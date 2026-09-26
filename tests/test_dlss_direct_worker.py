from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from postprocessing.dlss5 import direct_session, direct_worker


class _OSNameProxy:
    """Override a module's platform name without changing pathlib globally."""

    def __init__(self, module, name):
        self._module = module
        self.name = name

    def __getattr__(self, name):
        return getattr(self._module, name)


class _Watchdog:
    def __init__(self, _parent_pid):
        pass

    def call(self, _operation, _timeout, function):
        return function()

    def close(self):
        pass


class _Native:
    def __init__(self, *_args):
        self.did_shutdown = False
        self.temporal = None

    def initialize(self):
        return "test-bridge", "test-gpu"

    def nvof_available(self):
        return True

    def nvof_info(self):
        return "NVOFA available"

    def process_frame(self, data, _width, _height, _reset, _intensity, temporal):
        self.temporal = temporal
        return data

    def shutdown(self):
        self.did_shutdown = True


def _control(op, request_id, **kwargs):
    return (json.dumps({"protocol": 1, "op": op, "request_id": request_id, **kwargs}) + "\n").encode()


class _MemoryPipe:
    def __init__(self, *, read_chunk=17):
        self._buffer = bytearray()
        self._condition = threading.Condition()
        self._writer_closed = False
        self._read_chunk = read_chunk

    def feed(self, data: bytes):
        with self._condition:
            self._buffer.extend(data)
            self._condition.notify_all()

    def close_writer(self):
        with self._condition:
            self._writer_closed = True
            self._condition.notify_all()

    def readline(self, limit=-1):
        if limit < 0:
            limit = 1 << 20
        with self._condition:
            while True:
                newline = self._buffer.find(b"\n")
                if newline >= 0:
                    count = min(newline + 1, limit)
                    break
                if len(self._buffer) >= limit:
                    count = limit
                    break
                if self._writer_closed:
                    if not self._buffer:
                        return b""
                    count = min(len(self._buffer), limit)
                    break
                self._condition.wait(0.05)
            result = bytes(self._buffer[:count])
            del self._buffer[:count]
            return result

    def readinto(self, target):
        with self._condition:
            while not self._buffer and not self._writer_closed:
                self._condition.wait(0.05)
            if not self._buffer:
                return 0
            count = min(len(target), len(self._buffer), self._read_chunk)
            target[:count] = self._buffer[:count]
            del self._buffer[:count]
            return count

    def read(self, size=-1):
        if size < 0:
            size = 4096
        result = bytearray(min(size, self._read_chunk))
        count = self.readinto(result)
        return bytes(result[:count])

    def close(self):
        self.close_writer()


class _FakeStdin:
    def __init__(self, process):
        self.process = process

    def write(self, data):
        data = bytes(data[:7])  # exercise the parent's partial-write loop
        self.process.receive(data)
        return len(data)

    def flush(self):
        pass

    def close(self):
        self.process.stdout.close_writer()


class _BlockAfterControlStdin:
    """Accept one JSON request, then simulate a worker that never reads pixels."""

    def __init__(self, process):
        self.process = process
        self._control = bytearray()
        self._control_done = False

    def write(self, data):
        if self._control_done:
            self.process._stopped.wait()
            raise BrokenPipeError("worker stopped consuming frame data")
        chunk = bytes(data[:7])
        self._control.extend(chunk)
        self.process.receive(chunk)
        if b"\n" in self._control:
            self._control_done = True
        return len(chunk)

    def flush(self):
        pass

    def close(self):
        self.process.stdout.close_writer()


class _FakePopen:
    """Small byte-stream worker double; it never loads a native DLL or GPU."""

    def __init__(self, _command, **_kwargs):
        self.pid = os.getpid()
        self.returncode = None
        self.stdout = _MemoryPipe(read_chunk=11)  # exercise partial payload reads
        self.stderr = _MemoryPipe()
        self.stdin = _FakeStdin(self)
        self._incoming = bytearray()
        self._payload_size = 0
        self._frame_request = None
        self.frames = 0
        self.operations = []
        self._stopped = threading.Event()
        self.mode = type(self).mode
        self.abort_on_frame = False

    mode = "normal"

    def receive(self, data):
        if self.returncode is not None:
            raise BrokenPipeError("worker exited")
        self._incoming.extend(data)
        while True:
            if self._payload_size:
                if len(self._incoming) < self._payload_size:
                    return
                payload = bytes(self._incoming[:self._payload_size])
                del self._incoming[:self._payload_size]
                request = self._frame_request
                self._payload_size = 0
                self._frame_request = None
                self.frames += 1
                self.operations.append(("frame", request["index"]))
                if self.mode == "stall_frame":
                    continue
                if self.mode == "nan_frame":
                    payload = np.full(len(payload) // 4, np.nan, dtype=np.float32).tobytes()
                if self.mode == "inf_frame":
                    payload = np.full(len(payload) // 4, np.inf, dtype=np.float32).tobytes()
                if self.mode == "unwritten_frame":
                    payload = b"\xff" * len(payload)
                if self.mode == "short_frame":
                    payload = payload[:-4]
                self.respond("frame", request["request_id"], index=request["index"], byte_count=len(payload))
                self.stdout.feed(payload)
                continue
            newline = self._incoming.find(b"\n")
            if newline < 0:
                return
            line = bytes(self._incoming[:newline + 1])
            del self._incoming[:newline + 1]
            request = json.loads(line)
            op = request["op"]
            request_id = request["request_id"]
            self.operations.append((op, request_id))
            if op == "init":
                status = "error" if self.mode == "no_nvof" and request["temporal"] else "ok"
                fields = {"status": status, "worker_pid": self.pid, "frames": request["frames"],
                          "bridge_version": "test", "gpu": "test", "nvof_available": self.mode != "no_nvof",
                          "nvof_info": "NVOFA missing" if self.mode == "no_nvof" else "NVOFA available"}
                if status == "error":
                    fields["error"] = "Temporal direct NR requires NVOFA"
                self.respond("ready", request_id, **fields)
            elif op == "frame":
                self._payload_size = request["byte_count"]
                self._frame_request = request
            elif op == "finish":
                if self.mode == "bad_finish":
                    self.respond("finished", request_id, status="ok", frames_processed=self.frames - 1,
                                 frames_expected=self.frames)
                else:
                    self.respond("finished", request_id, status="ok", frames_processed=self.frames,
                                 frames_expected=self.frames)
            elif op == "close":
                if self.mode not in {"stall_shutdown", "unkillable_shutdown"}:
                    self.respond("closed", request_id, status="ok", native_shutdown=True,
                                 frames_processed=self.frames, frames_expected=self.frames)
                    self.exit(0)

    def respond(self, op, request_id, **fields):
        self.stdout.feed(_control(op, request_id, **fields))

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            if timeout is not None:
                raise subprocess.TimeoutExpired("fake-direct-worker", timeout)
            self.returncode = 0
        return self.returncode

    def exit(self, code):
        self.returncode = code
        self._stopped.set()
        self.stdout.close_writer()
        self.stderr.close_writer()

    def kill(self):
        if self.mode == "unkillable_shutdown":
            return
        self.exit(124)


class DirectWorkerServeTests(unittest.TestCase):
    def test_native_output_buffer_starts_nonfinite_to_expose_partial_writes(self):
        class LeaveOutputUnwritten:
            def __call__(self, *_args):
                return 1

        native = direct_worker.NativeBridge.__new__(direct_worker.NativeBridge)
        native.lib = SimpleNamespace(dlss5nr_process=LeaveOutputUnwritten())
        result = native.process_frame(np.zeros(12, dtype=np.float32).tobytes(), 2, 2, True, 1.0, False)
        self.assertFalse(np.isfinite(np.frombuffer(result, dtype=np.float32)).all())

    def test_watchdog_rechecks_completed_operation_after_waking(self):
        terminated = []
        watchdog = direct_worker.OperationWatchdog(
            None, terminate=terminated.append, interval=0.08,
        )
        try:
            time.sleep(0.01)  # let the watcher enter its condition wait
            self.assertEqual(watchdog.call("quick operation", 0.01, lambda: "done"), "done")
            time.sleep(0.1)  # past the operation's old deadline
            self.assertEqual(terminated, [])
        finally:
            watchdog.close()

    def _runtime(self, root: Path):
        native_runtime = root / "runtime"
        caller = native_runtime / "caller"
        bridge = root / "native" / "bin" / "dlss5nr_bridge.dll"
        caller.mkdir(parents=True)
        bridge.parent.mkdir(parents=True)
        (native_runtime / "nvngx_dlssnr.dll").write_bytes(b"test")
        (caller / "nvngx.dll_comfy.dll").write_bytes(b"test")
        bridge.write_bytes(b"test")
        return native_runtime, bridge

    def _message(self, op, request_id, **kwargs):
        return (json.dumps({"protocol": 1, "op": op, "request_id": request_id, **kwargs}) + "\n").encode()

    def test_temporal_frame_requires_nvof_and_finish_receipt_precedes_shutdown(self):
        if os.name != "nt":
            self.skipTest("the configured runtime path validation is Windows-specific")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native_runtime, bridge = self._runtime(root)
            init = self._message("init", 1, runtime_dir=str(native_runtime), bridge_path=str(bridge),
                                 width=1, height=1, frames=1, temporal=True, intensity=1.0)
            rgb = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
            frame = self._message("frame", 2, index=0, reset=True, byte_count=len(rgb)) + rgb
            finish = self._message("finish", 3)
            close = self._message("close", 4)
            native_instances = []

            def make_native(*args):
                result = _Native(*args)
                native_instances.append(result)
                return result

            output = io.BytesIO()
            code, loaded = direct_worker.serve(
                io.BytesIO(init + frame + finish + close), output,
                parent_pid=None, stderr=io.BytesIO(), native_factory=make_native,
                watchdog_factory=_Watchdog,
            )
            self.assertEqual((code, loaded), (0, True))
            response_stream = io.BytesIO(output.getvalue())
            responses = []
            while line := response_stream.readline():
                responses.append(json.loads(line))
                if responses[-1]["op"] == "frame":
                    self.assertEqual(response_stream.read(responses[-1]["byte_count"]), rgb)
            self.assertEqual([item["op"] for item in responses], ["ready", "frame", "finished", "closed"])
            self.assertEqual(responses[-2]["frames_processed"], 1)
            self.assertTrue(responses[-1]["native_shutdown"])
            self.assertTrue(native_instances[0].did_shutdown)
            self.assertIs(native_instances[0].temporal, True)

    def test_worker_rejects_premature_finish_without_shutdown(self):
        if os.name != "nt":
            self.skipTest("the configured runtime path validation is Windows-specific")
        with tempfile.TemporaryDirectory() as temp:
            native_runtime, bridge = self._runtime(Path(temp))
            init = self._message("init", 1, runtime_dir=str(native_runtime), bridge_path=str(bridge),
                                 width=1, height=1, frames=1, temporal=False, intensity=1.0)
            output = io.BytesIO()
            native_instances = []

            def make_native(*args):
                result = _Native(*args)
                native_instances.append(result)
                return result

            code, _ = direct_worker.serve(
                io.BytesIO(init + self._message("finish", 2)), output,
                parent_pid=None, stderr=io.BytesIO(), native_factory=make_native,
                watchdog_factory=_Watchdog,
            )
            self.assertEqual(code, direct_worker.EXIT_WORKER_ERROR)
            response = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual(response["status"], "error")
            self.assertIn("processed 0 of 1", response["error"])
            self.assertFalse(native_instances[0].did_shutdown)

    def test_invalid_native_output_length_is_reported_as_frame_error(self):
        class BadNative(_Native):
            def process_frame(self, *_args):
                return b""

        if os.name != "nt":
            self.skipTest("the configured runtime path validation is Windows-specific")
        with tempfile.TemporaryDirectory() as temp:
            native_runtime, bridge = self._runtime(Path(temp))
            init = self._message("init", 1, runtime_dir=str(native_runtime), bridge_path=str(bridge),
                                 width=1, height=1, frames=1, temporal=False, intensity=1.0)
            rgb = np.zeros(3, dtype=np.float32).tobytes()
            frame = self._message("frame", 2, index=0, reset=True, byte_count=len(rgb)) + rgb
            output = io.BytesIO()
            code, _ = direct_worker.serve(
                io.BytesIO(init + frame), output, parent_pid=None, stderr=io.BytesIO(),
                native_factory=BadNative, watchdog_factory=_Watchdog,
            )
            self.assertEqual(code, direct_worker.EXIT_WORKER_ERROR)
            response = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual(response["op"], "frame")
            self.assertEqual(response["status"], "error")
            self.assertIn("invalid RGB output length", response["error"])


class DirectNRSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "runtime" / "caller").mkdir(parents=True)
        (self.root / "runtime" / "nvngx_dlssnr.dll").write_bytes(b"fake")
        (self.root / "runtime" / "caller" / "nvngx.dll_comfy.dll").write_bytes(b"fake")
        (self.root / "native" / "bin").mkdir(parents=True)
        (self.root / "native" / "bin" / "dlss5nr_bridge.dll").write_bytes(b"fake")
        self.popen_patch = mock.patch.object(direct_session.subprocess, "Popen", _FakePopen)
        self.popen_patch.start()
        self.addCleanup(self.popen_patch.stop)
        _FakePopen.mode = "normal"

    def _session(self, *, frames=1, temporal=False, abort_callback=None, width=2, height=1):
        # The session process protocol is platform independent. Simulate only
        # its module-local Windows gate so the mocked protocol suite also runs
        # on Linux CI; pathlib and the rest of Python retain the host platform.
        with mock.patch.object(direct_session, "os", _OSNameProxy(os, "nt")):
            return direct_session.DirectNRSession(
                width, height, frames, temporal=temporal, intensity=1.0,
                runtime_dir=self.root, abort_callback=abort_callback,
            )

    def test_non_windows_gate_refuses_native_launch(self):
        with (
            mock.patch.object(direct_session, "os", _OSNameProxy(os, "posix")),
            mock.patch.object(direct_session.subprocess, "Popen") as popen,
            self.assertRaisesRegex(direct_session.DirectNRSessionError, "Windows-only"),
        ):
            direct_session.DirectNRSession(
                2, 1, 1, temporal=False, intensity=1.0, runtime_dir=self.root,
            )
        popen.assert_not_called()

    def test_streams_frames_and_accepts_only_finished_then_closed(self):
        session = self._session()
        source = np.array([[[10, 20, 30, 4], [80, 90, 100, 7]]], dtype=np.uint8)
        output = np.zeros_like(source)
        returned = session.process_frame(0, source, True, output)
        self.assertIs(returned, output)
        self.assertTrue(np.array_equal(returned, source))
        self.assertTrue(session.close())
        self.assertTrue(session.render_finished)
        self.assertTrue(session.native_shutdown)
        self.assertEqual(session.cleanup_status, "native_shutdown")

    def test_premature_close_is_rejected_and_does_not_claim_render_success(self):
        session = self._session(frames=2)
        with self.assertRaisesRegex(direct_session.DirectNRSessionError, "processed 0 of 2"):
            session.close()
        self.assertFalse(session.render_finished)
        self.assertEqual(session.cleanup_status, "incomplete")
        self.assertFalse(any(op == "finish" for op, _ in session._process.operations))

    def test_temporal_mode_refuses_missing_nvof(self):
        _FakePopen.mode = "no_nvof"
        with self.assertRaisesRegex(direct_session.DirectNRSessionError, "requires NVOFA"):
            self._session(temporal=True)

    def test_invalid_length_nonfinite_and_unwritten_pixels_abort_session(self):
        source = np.zeros((1, 2, 4), dtype=np.uint8)
        for mode in ("short_frame", "nan_frame", "inf_frame", "unwritten_frame"):
            with self.subTest(mode=mode):
                _FakePopen.mode = mode
                session = self._session()
                with self.assertRaises(direct_session.DirectNRSessionError):
                    session.process_frame(0, source, True)
                self.assertTrue(session._failed)
                session.close(abort=True)

    def test_float_input_cannot_cross_the_trusted_parent_boundary(self):
        for value in (0.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                session = self._session()
                try:
                    source = np.full((1, 2, 4), value, dtype=np.float32)
                    with self.assertRaisesRegex(ValueError, "uint8 RGBA"):
                        session.process_frame(0, source, True)
                    self.assertFalse(any(op == "frame" for op, _ in session._process.operations))
                finally:
                    session.close(abort=True)

    def test_cancellation_terminates_worker_without_a_finish_receipt(self):
        _FakePopen.mode = "stall_frame"
        session = self._session()
        source = np.zeros((1, 2, 4), dtype=np.uint8)
        calls = 0

        def delayed_abort():
            nonlocal calls
            calls += 1
            return calls >= 4

        session.abort_callback = delayed_abort
        with self.assertRaises(direct_session.DirectNRCancelled):
            session.process_frame(0, source, True)
        self.assertFalse(session.render_finished)
        self.assertEqual(session.cleanup_status, "aborted")

    def test_blocked_large_payload_write_is_bounded_by_frame_deadline(self):
        session = self._session(width=1024, height=1024)
        session._process.stdin = _BlockAfterControlStdin(session._process)
        source = np.zeros((1024, 1024, 4), dtype=np.uint8)
        with mock.patch.object(direct_session, "FRAME_TIMEOUT_SECONDS", 0.08):
            with self.assertRaises(direct_session.DirectNRTimeoutError):
                session.process_frame(0, source, True)
        self.assertIsNotNone(session._process.poll())
        self.assertTrue(session._failed)

    def test_shutdown_timeout_preserves_validated_render_completion(self):
        _FakePopen.mode = "stall_shutdown"
        session = self._session()
        session.process_frame(0, np.zeros((1, 2, 4), dtype=np.uint8), True)
        with mock.patch.object(direct_session, "SHUTDOWN_TIMEOUT_SECONDS", 0.08):
            self.assertFalse(session.close())
        self.assertTrue(session.render_finished)
        self.assertFalse(session.native_shutdown)
        self.assertEqual(session.cleanup_status, "forced_after_finish")

    def test_shutdown_worker_that_cannot_be_terminated_raises(self):
        _FakePopen.mode = "unkillable_shutdown"
        session = self._session()
        session.process_frame(0, np.zeros((1, 2, 4), dtype=np.uint8), True)
        with mock.patch.object(direct_session, "SHUTDOWN_TIMEOUT_SECONDS", 0.08):
            with self.assertRaisesRegex(direct_session.DirectNRSessionError, "could not be terminated"):
                session.close()
        self.assertTrue(session.render_finished)
        self.assertEqual(session.cleanup_status, "termination_failed_after_finish")

    def test_worker_with_bad_finish_receipt_is_not_accepted_as_complete(self):
        _FakePopen.mode = "bad_finish"
        session = self._session()
        session.process_frame(0, np.zeros((1, 2, 4), dtype=np.uint8), True)
        with self.assertRaisesRegex(direct_session.DirectNRSessionError, "finish receipt"):
            session.close()
        self.assertFalse(session.render_finished)
        self.assertEqual(session.cleanup_status, "finish_failed")

    def test_dimensions_and_frame_contract_are_checked_before_native_work(self):
        with self.assertRaises(ValueError):
            direct_session.DirectNRSession(0, 1, 1, temporal=False, intensity=1, runtime_dir=self.root)
        session = self._session()
        with self.assertRaises(ValueError):
            session.process_frame(1, np.zeros((1, 2, 4), dtype=np.uint8), True)
        with self.assertRaises(ValueError):
            session.process_frame(0, np.zeros((1, 2, 3), dtype=np.uint8), True)
        session.close(abort=True)


if __name__ == "__main__":
    unittest.main()
