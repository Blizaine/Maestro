"""Long-video mask composition keeps temporary memory independent of duration."""
import ast
from pathlib import Path
import tracemalloc
import unittest

import numpy as np


class TestRecastMaskComposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "app/launch.py"
        node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_compose_recast_character_masks")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        cls.compose = staticmethod(namespace[node.name])

    def test_priority_background_and_global_weighted_overlap(self):
        first = np.zeros((2, 4, 8, 3), dtype=np.uint8)
        second = np.zeros_like(first)
        first[0, :2, :2] = 255
        second[0, :2, :2] = 255  # Complete overlap in this frame.
        second[1, :, :] = 255  # No overlap in the much larger second-frame area.
        merged, overlaps = self.compose([first, second], [(255, 0, 0), (0, 255, 0)])
        np.testing.assert_array_equal(merged[0, 0, 0], [255, 0, 0])
        np.testing.assert_array_equal(merged[0, 3, 7], [255, 255, 255])
        np.testing.assert_array_equal(merged[1, 3, 7], [0, 255, 0])
        self.assertAlmostEqual(overlaps[1], 4 / 36)
        with self.assertRaisesRegex(ValueError, "overlaps an earlier mapping"):
            self.compose([first, second], [(255, 0, 0), (0, 255, 0)], overlap_limit=0.1)

    def test_binary_video_and_single_rgb_frame(self):
        binary = np.zeros((2, 4, 8), dtype=bool)
        binary[:, 0, 0] = True
        merged, overlaps = self.compose([binary], [(12, 34, 56)], background_color=(0, 0, 0))
        self.assertEqual(merged.shape, (2, 4, 8, 3))
        np.testing.assert_array_equal(merged[1, 0, 0], [12, 34, 56])
        single = np.repeat(binary[0, ..., None], 3, axis=-1).astype(np.uint8) * 255
        image, _ = self.compose([single], [(12, 34, 56)], background_color=(0, 0, 0))
        np.testing.assert_array_equal(merged[0], image)
        self.assertEqual(overlaps, [0.0])

    def test_rejects_empty_and_mismatched_masks(self):
        with self.assertRaisesRegex(ValueError, "matched no pixels"):
            self.compose([np.zeros((4, 8, 3), dtype=np.uint8)], [(255, 0, 0)])
        with self.assertRaisesRegex(ValueError, "matching dimensions"):
            self.compose([np.ones((4, 8, 3)), np.ones((8, 8, 3))], [(255, 0, 0), (0, 255, 0)])

    def test_long_sequence_allocates_only_output_plus_frame_scratch(self):
        # Broadcast views model an arbitrarily long input without retaining
        # additional test copies. NumPy reports owned buffers to tracemalloc.
        a = np.zeros((48, 64, 3), dtype=np.uint8)
        b = np.zeros_like(a)
        a[:, :32] = 255
        b[:, 32:] = 255
        masks = [np.broadcast_to(frame, (1000, *frame.shape)) for frame in (a, b)]
        tracemalloc.start()
        try:
            merged, overlaps = self.compose(masks, [(255, 0, 0), (0, 255, 0)])
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, merged.nbytes + 1024 * 1024,
                        f"Scratch allocation grew with the full video: {peak} bytes")
        self.assertEqual(overlaps, [0.0, 0.0])
        np.testing.assert_array_equal(merged[-1, 0, 0], [255, 0, 0])
        np.testing.assert_array_equal(merged[-1, 0, -1], [0, 255, 0])


if __name__ == "__main__":
    unittest.main()
