"""CPU-only contracts for the Windows 11 DLSS-NR alignment adapter."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1] / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from postprocessing.dlss5.alignment import (
    AlignedNeuralSession,
    _even_output_dimension,
    _next_aligned_source,
)


def _even_output(source: int, scale: float) -> int:
    return max(2, math.floor(source * scale / 2 + 0.5) * 2)


class FakeNeuralWorker:
    def __init__(
        self,
        width,
        height,
        frames,
        scale,
        intensity,
        abort_callback,
        *,
        render_override=None,
        output_override=None,
        process_error=None,
        wrong_frame_shape=False,
    ):
        self.render_width, self.render_height = render_override or (width, height)
        self.output_width, self.output_height = output_override or (
            _even_output(width, scale),
            _even_output(height, scale),
        )
        self.frames = frames
        self.scale = scale
        self.intensity = intensity
        self.abort_callback = abort_callback
        self.needs_guides = True
        self.process_error = process_error
        self.wrong_frame_shape = wrong_frame_shape
        self.process_calls = []
        self.close_calls = []

    def process_frame(self, index, rgba, motion, reset, output=None, *, depth=None):
        self.process_calls.append((index, rgba, motion, reset, output, depth))
        if self.process_error is not None:
            raise self.process_error
        width = self.output_width - int(self.wrong_frame_shape)
        height = self.output_height
        result = np.empty((height, width, 4), dtype=np.uint8)
        result[..., 0] = np.arange(width, dtype=np.uint16) % 256
        result[..., 1] = np.arange(height, dtype=np.uint16)[:, None] % 256
        result[..., 2] = index
        result[..., 3] = 255
        if output is not None and result.shape == output.shape:
            np.copyto(output, result)
            return output
        return result

    def close(self, *, abort=False):
        self.close_calls.append(abort)


class FakeFactory:
    def __init__(self, **worker_options):
        self.worker_options = worker_options
        self.calls = []
        self.workers = []

    def __call__(self, width, height, frames, scale, intensity, abort_callback):
        self.calls.append((width, height, frames, scale, intensity, abort_callback))
        worker = FakeNeuralWorker(
            width,
            height,
            frames,
            scale,
            intensity,
            abort_callback,
            **self.worker_options,
        )
        self.workers.append(worker)
        return worker


class AlignedNeuralSessionGeometryTests(unittest.TestCase):
    def test_bounded_search_covers_supported_sizes_without_midrange_cutoffs(self):
        expected_counts = {
            1.0: (7680, 0),
            1.5: (5120, 0),
            1.724: (4419, 36),
            2.0: (3840, 0),
            3.0: (2560, 0),
        }
        for scale, (expected_successes, expected_failures) in expected_counts.items():
            with self.subTest(scale=scale):
                successes = 0
                failures = []
                for source in range(1, 7681):
                    if _even_output_dimension(source, scale) > 7680:
                        continue
                    try:
                        padded, native_output = _next_aligned_source(
                            source, scale, "width"
                        )
                    except ValueError:
                        failures.append(source)
                        continue

                    successes += 1
                    self.assertGreaterEqual(padded, source)
                    if _even_output_dimension(source, scale) % 64 == 0:
                        self.assertEqual(padded, source)
                    else:
                        self.assertEqual(padded % 2, 0)
                    self.assertLessEqual(padded - source, 128)
                    self.assertLessEqual(native_output, 7680)
                    self.assertEqual(native_output % 64, 0)
                    self.assertEqual(native_output, _even_output_dimension(padded, scale))
                    first_candidate = source + (source & 1)
                    for earlier in range(first_candidate, padded, 2):
                        self.assertNotEqual(
                            _even_output_dimension(earlier, scale) % 64,
                            0,
                            msg=f"{scale=} {source=} skipped an earlier aligned candidate",
                        )

                self.assertEqual(successes, expected_successes)
                self.assertEqual(len(failures), expected_failures)
                if scale == 1.724:
                    self.assertEqual(failures, list(range(4419, 4455)))
                else:
                    self.assertEqual(failures, [])

    def test_issue_geometries_cover_both_orientations_and_supported_scales(self):
        # (source, scale, padded source, native output, public output)
        cases = (
            ((704, 1280), 1.0, (704, 1280), (704, 1280), (704, 1280)),
            ((1280, 704), 1.0, (1280, 704), (1280, 704), (1280, 704)),
            ((704, 1280), 1.5, (768, 1280), (1152, 1920), (1056, 1920)),
            ((1280, 704), 1.5, (1280, 768), (1920, 1152), (1920, 1056)),
            ((704, 1280), 1.724, (742, 1336), (1280, 2304), (1214, 2206)),
            ((1280, 704), 1.724, (1336, 742), (2304, 1280), (2206, 1214)),
            ((704, 1280), 2.0, (704, 1280), (1408, 2560), (1408, 2560)),
            ((1280, 704), 2.0, (1280, 704), (2560, 1408), (2560, 1408)),
            ((704, 1280), 3.0, (704, 1280), (2112, 3840), (2112, 3840)),
            ((1280, 704), 3.0, (1280, 704), (3840, 2112), (3840, 2112)),
            ((720, 720), 1.0, (768, 768), (768, 768), (720, 720)),
            ((512, 512), 1.5, (512, 512), (768, 768), (768, 768)),
        )
        for source, scale, padded, native_output, public_output in cases:
            with self.subTest(source=source, scale=scale):
                factory = FakeFactory()
                session = AlignedNeuralSession(
                    *source, 3, scale, session_factory=factory
                )
                self.assertEqual(factory.calls[0][:2], padded)
                self.assertEqual(
                    (factory.workers[0].output_width, factory.workers[0].output_height),
                    native_output,
                )
                self.assertEqual((session.render_width, session.render_height), source)
                self.assertEqual((session.output_width, session.output_height), public_output)
                self.assertEqual(
                    (session.native_render_width, session.native_render_height), padded
                )
                session.close(abort=True)

    def test_edge_padding_preserves_content_and_crops_into_the_callers_buffer(self):
        width, height = 144, 160
        factory = FakeFactory()
        session = AlignedNeuralSession(width, height, 2, 1.5, session_factory=factory)

        x = np.arange(width, dtype=np.uint16)[None, :]
        y = np.arange(height, dtype=np.uint16)[:, None]
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = x % 256
        rgba[..., 1] = y % 256
        rgba[..., 2] = (x + y) % 256
        rgba[..., 3] = 255
        motion = np.empty((height, width, 2), dtype=np.float16)
        motion[..., 0] = x
        motion[..., 1] = y
        depth = (x + y).astype(np.float32)
        output = np.full((session.output_height, session.output_width, 4), 7, dtype=np.uint8)

        result = session.process_frame(0, rgba, motion, True, output, depth=depth)
        self.assertIs(result, output)
        self.assertEqual(result.shape, (240, 216, 4))
        self.assertTrue(result.flags.c_contiguous)
        self.assertEqual(result[0, -1, 0], 215)
        self.assertEqual(result[-1, 0, 1], 239)

        _index, padded_rgba, padded_motion, _reset, worker_output, padded_depth = (
            factory.workers[0].process_calls[0]
        )
        self.assertIsNone(worker_output)
        self.assertEqual(padded_rgba.shape, (170, 170, 4))
        self.assertEqual(padded_motion.shape, (170, 170, 2))
        self.assertEqual(padded_depth.shape, (170, 170))
        np.testing.assert_array_equal(padded_rgba[:height, :width], rgba)
        np.testing.assert_array_equal(padded_motion[:height, :width], motion)
        np.testing.assert_array_equal(padded_depth[:height, :width], depth)
        np.testing.assert_array_equal(
            padded_rgba[:height, width:],
            np.broadcast_to(rgba[:, -1:, :], padded_rgba[:height, width:].shape),
        )
        np.testing.assert_array_equal(
            padded_rgba[height:, :, :],
            np.broadcast_to(padded_rgba[-1:, :, :], padded_rgba[height:, :, :].shape),
        )
        np.testing.assert_array_equal(
            padded_motion[:height, width:],
            np.broadcast_to(motion[:, -1:, :], padded_motion[:height, width:].shape),
        )
        np.testing.assert_array_equal(
            padded_depth[height:, :],
            np.broadcast_to(padded_depth[-1:, :], padded_depth[height:, :].shape),
        )

        unbuffered = session.process_frame(1, rgba, motion, False, depth=depth)
        self.assertEqual(unbuffered.shape, (240, 216, 4))
        self.assertTrue(unbuffered.flags.c_contiguous)
        self.assertEqual(unbuffered[0, -1, 0], 215)
        self.assertEqual(unbuffered[-1, 0, 1], 239)
        session.close()

    def test_aligned_worker_uses_requested_dimensions_and_output_buffer_directly(self):
        factory = FakeFactory()
        session = AlignedNeuralSession(512, 512, 1, 1.5, session_factory=factory)
        worker = factory.workers[0]
        rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        motion = np.zeros((512, 512, 2), dtype=np.float16)
        depth = np.zeros((512, 512), dtype=np.float32)
        output = np.empty((768, 768, 4), dtype=np.uint8)

        result = session.process_frame(0, rgba, motion, True, output, depth=depth)

        self.assertEqual(factory.calls[0][:2], (512, 512))
        self.assertIs(worker.process_calls[0][1], rgba)
        self.assertIs(worker.process_calls[0][2], motion)
        self.assertIs(worker.process_calls[0][4], output)
        self.assertIs(result, output)
        self.assertEqual(result[-1, -1, 0], (768 - 1) % 256)
        session.close()

    def test_odd_source_keeps_the_existing_even_rounded_public_output(self):
        factory = FakeFactory()
        session = AlignedNeuralSession(101, 99, 1, 1.0, session_factory=factory)
        self.assertEqual((session.output_width, session.output_height), (102, 100))
        self.assertEqual(factory.calls[0][:2], (128, 128))
        self.assertEqual(
            (factory.workers[0].output_width, factory.workers[0].output_height),
            (128, 128),
        )
        session.close()

    def test_odd_source_with_aligned_output_is_passed_through_without_padding(self):
        factory = FakeFactory()
        with self.assertNoLogs("postprocessing.dlss5.alignment", level="INFO"):
            session = AlignedNeuralSession(63, 63, 1, 1.0, session_factory=factory)
        self.assertEqual((session.output_width, session.output_height), (64, 64))
        self.assertEqual(factory.calls[0][:2], (63, 63))
        self.assertEqual((session.native_render_width, session.native_render_height), (63, 63))
        session.close()

    def test_padded_session_logs_requested_and_native_geometry(self):
        factory = FakeFactory()
        with self.assertLogs("postprocessing.dlss5.alignment", level="INFO") as captured:
            session = AlignedNeuralSession(704, 1280, 1, 1.5, session_factory=factory)
        self.assertIn(
            "[DLSS Win11 alignment] 704x1280 -> requested 1056x1920; "
            "internal 768x1280 -> native 1152x1920 (scale=1.5)",
            captured.output[0],
        )
        session.close()

    def test_geometry_limits_and_impossible_alignment_fail_before_worker_start(self):
        invalid = (
            (0, 100, 1.5),
            (100, -1, 1.5),
            (100.5, 100, 1.5),
            (100, 100, math.nan),
            (100, 100, math.inf),
            (100, 100, 1.25),
            (7681, 1, 1.0),
            (3000, 1000, 3.0),  # Requested output exceeds the long-side limit.
            (4320, 4320, 1.0),  # Required padding exceeds the short-side limit.
            (4419, 2000, 1.724),  # 7618px request cannot align before 7680px.
        )
        for width, height, scale in invalid:
            with self.subTest(width=width, height=height, scale=scale):
                factory = FakeFactory()
                with self.assertRaises(ValueError):
                    AlignedNeuralSession(
                        width, height, 1, scale, session_factory=factory
                    )
                self.assertEqual(factory.calls, [])

        factory = FakeFactory()
        with self.assertRaisesRegex(ValueError, "frame count exceeds"):
            AlignedNeuralSession(1, 1, 1 << 32, 1.5, session_factory=factory)
        self.assertEqual(factory.calls, [])

    def test_worker_negotiation_mismatch_closes_worker_before_raising(self):
        factory = FakeFactory(output_override=(999, 768))
        with self.assertRaisesRegex(RuntimeError, "unexpected output dimensions"):
            AlignedNeuralSession(512, 512, 1, 1.5, session_factory=factory)
        self.assertEqual(factory.workers[0].close_calls, [True])

    def test_worker_and_frame_failures_abort_the_worker(self):
        rgba = np.zeros((160, 144, 4), dtype=np.uint8)
        motion = np.zeros((160, 144, 2), dtype=np.float16)
        depth = np.zeros((160, 144), dtype=np.float32)
        for factory in (
            FakeFactory(process_error=RuntimeError("native frame failed")),
            FakeFactory(wrong_frame_shape=True),
        ):
            with self.subTest(factory=factory.worker_options):
                session = AlignedNeuralSession(144, 160, 1, 1.5, session_factory=factory)
                with self.assertRaises(RuntimeError):
                    session.process_frame(0, rgba, motion, True, depth=depth)
                self.assertEqual(factory.workers[0].close_calls, [True])

    def test_bad_guides_abort_and_close_is_idempotent(self):
        factory = FakeFactory()
        session = AlignedNeuralSession(144, 160, 1, 1.5, session_factory=factory)
        with self.assertRaisesRegex(ValueError, "motion guide"):
            session.process_frame(
                0,
                np.zeros((160, 144, 4), dtype=np.uint8),
                np.zeros((159, 144, 2), dtype=np.float16),
                True,
            )
        self.assertEqual(factory.workers[0].close_calls, [True])
        session.close(abort=False)
        self.assertEqual(factory.workers[0].close_calls, [True])

    def test_factory_receives_abort_callback(self):
        abort_callback = lambda: False
        factory = FakeFactory()
        session = AlignedNeuralSession(
            512, 512, 2, 1.5, abort_callback=abort_callback, session_factory=factory
        )
        self.assertIs(factory.calls[0][-1], abort_callback)
        session.close()


if __name__ == "__main__":
    unittest.main()
