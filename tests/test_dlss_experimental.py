"""Windows 10 opt-in, routing and media contracts; no native GPU calls."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


class ExperimentalRoutingTests(unittest.TestCase):
    def test_opt_in_is_required_and_cannot_change_windows_11_or_frame_generation(self):
        from postprocessing.dlss5 import runtime
        if sys.platform != "win32":
            self.skipTest("Windows build routing")
        with tempfile.TemporaryDirectory() as folder, patch.object(runtime, "RUNTIME", Path(folder)), \
                patch.object(runtime.sys, "getwindowsversion", return_value=SimpleNamespace(build=19045)):
            self.assertFalse(runtime.uses_experimental_backend())
            marker = Path(folder) / "direct/enabled.json"
            marker.parent.mkdir()
            for content in ('{"enabled": false}', '{"enabled": "true"}', 'bad json', '[]'):
                marker.write_text(content)
                self.assertFalse(runtime.uses_experimental_backend())
            marker.write_text(json.dumps({"enabled": True}))
            self.assertTrue(runtime.uses_experimental_backend())
            self.assertIn("Windows 11", runtime.unavailable_reason(temporal=True))
            with patch.object(runtime.sys, "getwindowsversion", return_value=SimpleNamespace(build=22631)):
                self.assertFalse(runtime.uses_experimental_backend())

    def test_missing_native_install_stays_unavailable(self):
        from postprocessing.dlss5 import runtime
        with tempfile.TemporaryDirectory() as folder, patch.object(runtime, "RUNTIME", Path(folder)), \
                patch.object(runtime, "uses_experimental_backend", return_value=True):
            self.assertIn("missing experimental DLSS file", runtime.unavailable_reason(temporal=False))

    def test_experimental_capabilities_and_validation_share_runtime_modes(self):
        from postprocessing.dlss5 import runtime
        from services.media_processing import capabilities, validate_methods
        with patch.object(runtime, "unavailable_reason", return_value=""), \
                patch.object(runtime, "uses_experimental_backend", return_value=True), \
                patch.object(runtime, "dlssg_capabilities", return_value={}), \
                patch.object(runtime, "is_rtx_50_series", return_value=False):
            caps = capabilities()
        scales = list(runtime.NR_MODES)
        self.assertTrue(caps["neural_rendering"]["experimental"])
        self.assertEqual(caps["neural_rendering"]["scales"], scales)
        with patch("services.media_processing.capabilities", return_value=caps):
            for scale in scales:
                validate_methods(f"dlss5*{scale}")

    def test_backend_capability_can_narrow_allowed_runtime_modes(self):
        from services.media_processing import validate_methods
        caps = {"neural_rendering": {"available": True, "scales": [1., 2.], "reason": ""}}
        with patch("services.media_processing.capabilities", return_value=caps):
            validate_methods("dlss5*1")
            validate_methods("dlss5*2")
            for scale in (1.5, 1.724, 3):
                with self.assertRaisesRegex(ValueError, "supports scales"):
                    validate_methods(f"dlss5*{scale}")

    def test_experimental_fractional_and_x3_scales_use_sr_negotiated_dimensions(self):
        import numpy as np
        from postprocessing.dlss5.experimental import ExperimentalNeuralSession
        cases = (
            (1.5, True, 608, 352, 912, 528, 640, 960),
            (1.724, False, 608, 352, 1048, 606, 631, 1088),
            (3.0, False, 608, 352, 1824, 1056, 640, 1920),
        )
        for scale, still_image, width, height, output_width, output_height, sr_width, sr_output_width in cases:
            with self.subTest(scale=scale, still_image=still_image):
                right_padding = sr_width - width
                sr = Mock(render_width=sr_width, render_height=height,
                          output_width=sr_output_width, output_height=output_height)
                sr_output = np.zeros((output_height, sr_output_width, 4), dtype=np.uint8)
                sr_output[..., 0] = np.arange(sr_output_width, dtype=np.uint32) % 256
                sr.process_frame.return_value = sr_output
                nr = Mock()
                with patch("postprocessing.dlss5.experimental.installation_error", return_value=""), \
                        patch("postprocessing.dlss5.runtime.NeuralRenderingSession", return_value=sr) as factory, \
                        patch("postprocessing.dlss5.direct_session.DirectNRSession", return_value=nr) as direct:
                    session = ExperimentalNeuralSession(width, height, 2, scale, 1., None,
                        still_image=still_image, runtime_dir=Path("test/direct"))
                    self.assertEqual((session.output_width, session.output_height),
                                     (output_width, output_height))
                    self.assertTrue(session.needs_guides)
                    self.assertEqual(session.render_width, width)
                    self.assertEqual(session._right_padding, right_padding)
                    self.assertEqual(factory.call_args.args[:5], (sr_width, height, 2, scale, 1.))
                    self.assertEqual(direct.call_args.args[:3], (output_width, output_height, 2))
                    self.assertEqual(direct.call_args.kwargs["temporal"], not still_image)
                    rgba = np.arange(height * width * 4, dtype=np.uint8).reshape(height, width, 4)
                    motion = (np.arange(height * width * 2, dtype=np.int32) % 2048).astype(np.float16).reshape(height, width, 2)
                    depth = np.arange(height * width, dtype=np.float32).reshape(height, width)
                    frame = session.process_frame(0, rgba, motion, True, depth=depth)
                    padded_rgba, padded_motion = sr.process_frame.call_args.args[1:3]
                    padded_depth = sr.process_frame.call_args.kwargs["depth"]
                    self.assertEqual(padded_rgba.shape, (height, sr_width, 4))
                    self.assertEqual(padded_motion.shape, (height, sr_width, 2))
                    self.assertEqual(padded_depth.shape, (height, sr_width))
                    for source, padded in ((rgba, padded_rgba), (motion, padded_motion), (depth, padded_depth)):
                        np.testing.assert_array_equal(padded[:, :width], source)
                        np.testing.assert_array_equal(
                            padded[:, width:], np.repeat(source[:, -1:], right_padding, axis=1))
                    sr.process_frame.assert_called_once_with(0, padded_rgba, padded_motion, True, depth=padded_depth)
                    enhanced_input = nr.process_frame.call_args.args[1]
                    self.assertEqual(enhanced_input.shape, (output_height, output_width, 4))
                    np.testing.assert_array_equal(enhanced_input, sr_output[:, :output_width])
                    self.assertEqual(frame, nr.process_frame.return_value)
                    session.close()
                    sr.close.assert_called_once_with(abort=False)
                    nr.close.assert_called_once_with(abort=False)

    def test_x1_odd_geometry_and_aligned_x2_do_not_pad(self):
        from postprocessing.dlss5.experimental import ExperimentalNeuralSession
        with patch("postprocessing.dlss5.experimental.installation_error", return_value=""), \
                patch("postprocessing.dlss5.runtime.NeuralRenderingSession") as factory, \
                patch("postprocessing.dlss5.direct_session.DirectNRSession", return_value=Mock()) as direct:
            x1 = ExperimentalNeuralSession(607, 351, 1, 1., 1., None,
                still_image=True, runtime_dir=Path("test/direct"))
            self.assertEqual((x1.output_width, x1.output_height), (607, 351))
            self.assertEqual(x1._right_padding, 0)
            self.assertIsNone(x1.sr)
            self.assertEqual(direct.call_args.args[:3], (607, 351, 1))
            self.assertFalse(direct.call_args.kwargs["temporal"])
            factory.assert_not_called()

        sr = Mock(render_width=608, render_height=352, output_width=1216, output_height=704)
        with patch("postprocessing.dlss5.experimental.installation_error", return_value=""), \
                patch("postprocessing.dlss5.runtime.NeuralRenderingSession", return_value=sr) as factory, \
                patch("postprocessing.dlss5.direct_session.DirectNRSession", return_value=Mock()) as direct:
            x2 = ExperimentalNeuralSession(608, 352, 1, 2., 1., None,
                still_image=False, runtime_dir=Path("test/direct"))
            self.assertEqual(x2._right_padding, 0)
            self.assertEqual(factory.call_args.args[:4], (608, 352, 1, 2.))
            self.assertEqual(direct.call_args.args[:3], (1216, 704, 1))
            self.assertTrue(direct.call_args.kwargs["temporal"])

    def test_render_and_output_limits_are_checked_before_worker_startup(self):
        from postprocessing.dlss5.experimental import ExperimentalNeuralSession
        with patch("postprocessing.dlss5.experimental.installation_error", return_value=""), \
                patch("postprocessing.dlss5.runtime.NeuralRenderingSession") as factory, \
                patch("postprocessing.dlss5.direct_session.DirectNRSession") as direct:
            with self.assertRaisesRegex(ValueError, "output dimensions.*7680x4320"):
                ExperimentalNeuralSession(4000, 3000, 1, 2., 1., None,
                    still_image=True, runtime_dir=Path("test/direct"))
            factory.assert_not_called()
            direct.assert_not_called()

            # Original output is 4320x7680 and valid, but aligning its SR row
            # would exceed the native short-edge limit and must not start a worker.
            with self.assertRaisesRegex(ValueError, "padded SR output dimensions.*7680x4320"):
                ExperimentalNeuralSession(2880, 5120, 1, 1.5, 1., None,
                    still_image=True, runtime_dir=Path("test/direct"))
            factory.assert_not_called()
            direct.assert_not_called()

    def test_sr_dimension_mismatch_closes_worker_before_direct_startup(self):
        from postprocessing.dlss5.experimental import ExperimentalNeuralSession
        sr = Mock(render_width=640, render_height=352, output_width=958, output_height=528)
        with patch("postprocessing.dlss5.experimental.installation_error", return_value=""), \
                patch("postprocessing.dlss5.runtime.NeuralRenderingSession", return_value=sr), \
                patch("postprocessing.dlss5.direct_session.DirectNRSession") as direct:
            with self.assertRaisesRegex(RuntimeError, "negotiated unexpected dimensions"):
                ExperimentalNeuralSession(608, 352, 1, 1.5, 1., None,
                    still_image=False, runtime_dir=Path("test/direct"))
            sr.close.assert_called_once_with(abort=True)
            direct.assert_not_called()

    def test_experimental_rejects_scales_outside_runtime_modes_before_startup(self):
        from postprocessing.dlss5.experimental import ExperimentalNeuralSession
        with patch("postprocessing.dlss5.experimental.installation_error") as check_install:
            for scale in (1.25, 4., 0.):
                with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "supports x1"):
                    ExperimentalNeuralSession(5, 3, 1, scale, 1., None,
                        still_image=True, runtime_dir=Path("test/direct"))
        check_install.assert_not_called()

    def test_same_size_direct_does_not_load_engine_or_download_guides(self):
        from services.media_processing import prepare_dlss
        with patch("postprocessing.dlss5.runtime.uses_experimental_backend", return_value=True), \
                patch.dict(sys.modules, {"wgp": None}):
            prepare_dlss({"dlss_motion": "raft"}, neural=True, scale=1.)

    def test_x2_runs_real_sr_before_nr_and_closes_both(self):
        import numpy as np
        from postprocessing.dlss5.experimental import ExperimentalNeuralSession
        sr = Mock(render_width=608, render_height=352, output_width=1216, output_height=704)
        sr_output = np.zeros((704, 1216, 4), dtype=np.uint8)
        sr.process_frame.return_value = sr_output
        nr = Mock()
        with patch("postprocessing.dlss5.experimental.installation_error", return_value=""), \
                patch("postprocessing.dlss5.runtime.NeuralRenderingSession", return_value=sr) as factory, \
                patch("postprocessing.dlss5.direct_session.DirectNRSession", return_value=nr) as direct:
            session = ExperimentalNeuralSession(608, 352, 2, 2., 1., None, still_image=False, runtime_dir=Path("test/direct"))
            factory.assert_called_once()
            self.assertEqual(factory.call_args.kwargs["host_dir"], Path("test/direct/sr/host"))
            self.assertEqual(direct.call_args.args[:3], (1216, 704, 2))
            session.process_frame(0, "source", "motion", True, depth="depth")
            sr.process_frame.assert_called_once_with(0, "source", "motion", True, depth="depth")
            nr.process_frame.assert_called_once_with(0, sr_output, True, None)
            session.close()
            sr.close.assert_called_once_with(abort=False)
            nr.close.assert_called_once_with(abort=False)

    def test_sr_closed_when_direct_initialization_fails(self):
        from postprocessing.dlss5.experimental import ExperimentalNeuralSession
        sr = Mock(render_width=608, render_height=352, output_width=1216, output_height=704)
        with patch("postprocessing.dlss5.experimental.installation_error", return_value=""), \
                patch("postprocessing.dlss5.runtime.NeuralRenderingSession", return_value=sr), \
                patch("postprocessing.dlss5.direct_session.DirectNRSession", side_effect=RuntimeError("no NVOFA")):
            with self.assertRaisesRegex(RuntimeError, "no NVOFA"):
                ExperimentalNeuralSession(608, 352, 2, 2., 1., None, still_image=False, runtime_dir=Path("test"))
            sr.close.assert_called_once_with(abort=True)

    def test_tensor_roundtrip_preserves_dtype_channels_and_cancellation(self):
        import numpy as np
        import torch
        from postprocessing.dlss5 import runtime
        class Echo:
            output_width = render_width = 64
            output_height = render_height = 64
            needs_guides = False
            def __init__(self):
                self.closed = []
            def process_frame(self, index, frame, motion, reset, output, **kwargs):
                output[:] = frame
                return output
            def close(self, *, abort):
                self.closed.append(abort)
        for dtype in (torch.uint8, torch.float32):
            session = Echo()
            sample = torch.full((4, 1, 64, 64), 127, dtype=torch.uint8)
            if dtype != torch.uint8:
                sample = sample.float().div(127.5).sub(1)
            with patch.object(runtime, "require_runtime"), patch.object(runtime, "create_neural_session", return_value=session), \
                    patch.object(runtime, "DepthGuides", side_effect=AssertionError("no depth for x1")):
                result = runtime.neural_render(sample, 1., still_image=True, depth_resolution="half", motion_vector="raft")
                torch.testing.assert_close(result, sample)
                self.assertEqual(session.closed, [False])
                self.assertIsNone(runtime.neural_render(sample, 1., still_image=True, depth_resolution="half", motion_vector="raft", abort_callback=lambda: True))
                self.assertEqual(session.closed[-1], True)


if __name__ == "__main__":
    unittest.main()
