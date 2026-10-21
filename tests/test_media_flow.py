"""Timing, capability and cancellation regressions for the Wan2GP media port."""
import sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import unittest
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.media_processing import dlss_scale, normalize_options, validate_methods


def test_options_reject_invalid_and_nonfinite_values():
    for value in (-1, 2.1, float("nan"), float("inf")):
        with unittest.TestCase().assertRaisesRegex(ValueError, ""):

            normalize_options({"dlss_intensity": value})
    with unittest.TestCase().assertRaisesRegex(ValueError, ""):

        validate_methods(temporal="rife3", image=True, check_runtime=False)
    with unittest.TestCase().assertRaisesRegex(ValueError, ""):

        validate_methods(spatial="dlss5*4", check_runtime=False)
    assert dlss_scale("dlss5*1") == 1
    assert validate_methods("dlss5*3", "rife3", check_runtime=False)["dlss_depth"] == "half"


def test_dlss_factors_use_installed_worker_capability():
    caps = {"frame_generation": {"available": True, "factors": [2, 3, 4]}}
    with patch("services.media_processing.capabilities", return_value=caps):
        validate_methods(temporal="dlssg*4")
        with unittest.TestCase().assertRaisesRegex(ValueError, "RTX 50"):
            validate_methods(temporal="dlssg*5")


def test_rife_keeps_every_source_frame_after_frame_100():
    for factor in (2, 3, 4):
        import torch
        from postprocessing.rife.inference import process_frames
        class Interpolator:
            supports_timestep = True
            times = []
            def inference(self, left, right, timestep, scale):
                self.times.append(timestep)
                return left.lerp(right, timestep)
        model = Interpolator()
        frames = torch.linspace(-1, 1, 105).reshape(1, 105, 1, 1).expand(3, -1, 32, 32).clone()
        with patch("postprocessing.rife.inference.ssim_matlab", return_value=0.8):
            output = process_frames(model, "cpu", frames, multiplier=factor)
        assert output.shape[1] == (105 - 1) * factor + 1
        torch.testing.assert_close(output[:, ::factor], frames)
        assert model.times[:factor - 1] == [i / factor for i in range(1, factor)]


def test_rife_cancel_and_legacy_x3_rejection():
    import torch
    from postprocessing.rife.inference import process_frames
    frames = torch.zeros(3, 2, 32, 32)
    with unittest.TestCase().assertRaisesRegex(ValueError, "requires RIFE"):
        process_frames(object(), "cpu", frames, multiplier=3)
    assert process_frames(object(), "cpu", frames, multiplier=2, abort_callback=lambda: True) is None


def test_dlss_worker_probe_failure_is_not_reported_as_ready():
    from postprocessing.dlss5 import runtime
    if runtime.os.name != "nt":
        raise unittest.SkipTest("Windows capability path")
    with patch.object(runtime, "_missing", return_value=[]), patch.object(runtime, "_gpu_series", return_value=40), \
         patch.object(runtime.sys, "getwindowsversion", return_value=SimpleNamespace(build=22631)), patch.object(runtime, "_hags_enabled", return_value=True), patch.object(runtime, "dlssg_capabilities", return_value={}):
        assert "did not report" in runtime.unavailable_reason(temporal=True)


def test_native_worker_protocol_roundtrip_and_cleanup():
    """Exercise the pipe transport without requiring proprietary DLSS binaries."""
    from postprocessing.dlss5.runtime import Worker
    code = "import sys; data=sys.stdin.buffer.read(6); sys.stdout.buffer.write(data[::-1]); sys.stdout.buffer.flush()"
    worker = Worker([sys.executable, "-u", "-c", code], Path.cwd())
    try:
        worker.write(b"abcdef")
        assert worker.read_exact(6, "echo test") == b"fedcba"
    finally:
        worker.close()
    assert worker.process.poll() == 0
    assert worker.process.stdin.closed and worker.process.stdout.closed


def test_native_worker_cancellation_unblocks_pending_reply():
    from postprocessing.dlss5.runtime import Worker
    cancelled = threading.Event()
    worker = Worker([sys.executable, "-u", "-c", "import time; time.sleep(60)"], Path.cwd(), cancelled.is_set)
    timer = threading.Timer(0.4, cancelled.set)
    timer.start()
    started = time.monotonic()
    try:
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "exited before replying"):
            worker.read_exact(4, "cancel test")
        assert time.monotonic() - started < 10
        assert worker.process.poll() is not None
    finally:
        timer.cancel()
        worker.close(abort=True)


def test_generation_finishing_preserves_source_on_failure_and_cancellation():
    import tempfile
    from services.media_flow import ProcessingCancelled, process_generation_video

    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "generated.mp4"
        destination = Path(folder) / "generated.media-flow.mp4"
        source.write_bytes(b"generated-original")

        with patch("services.media_flow.process_video", side_effect=RuntimeError("native eval failed")):
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "native eval failed"):
                process_generation_video(source, destination, spatial="dlss5*1")
        assert source.read_bytes() == b"generated-original"
        assert not destination.exists()

        def render(_source, output, **_kwargs):
            Path(output).write_bytes(b"finished")
            return {"frames": 1}

        with patch("services.media_flow.process_video", side_effect=render):
            result = process_generation_video(source, destination, spatial="dlss5*1")
        assert result == {"frames": 1}
        assert source.read_bytes() == b"finished"
        assert not destination.exists()

        source.write_bytes(b"generated-original")
        with patch("services.media_flow.process_video", side_effect=render):
            with unittest.TestCase().assertRaisesRegex(ProcessingCancelled, "cancelled"):
                process_generation_video(source, destination, abort=lambda: True)
        assert source.read_bytes() == b"generated-original"
        assert not destination.exists()


def test_dlss_user_processing_only_uses_cached_readiness():
    with patch("services.media_processing._nr_modes", return_value=[1.0]), \
            patch("services.media_processing.capabilities", return_value={
        "neural_rendering": {
            "available": False,
            "reason": "readiness check has not run; use Refresh while idle",
            "scales": [1.0],
        },
        "frame_generation": {"available": True, "factors": [2]},
    }) as caps:
        with unittest.TestCase().assertRaisesRegex(ValueError, "use Refresh while idle"):
            validate_methods("dlss5*1")
    caps.assert_called_once_with(probe_neural=False)


def load_tests(loader, suite, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items() if name.startswith("test_") and callable(value))

if __name__ == "__main__":
    unittest.main()
