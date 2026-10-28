"""CPU-only geometry regressions for MiniMax H3 keyframe fitting."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

from PIL import Image, ImageDraw


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
_PACKING_PATH = _APP / "models" / "minimax_h3" / "packing.py"
_MAIN_PATH = _APP / "models" / "minimax_h3" / "minimax_h3_main.py"
_VIGGLE_PATH = _APP / "models" / "minimax_h3" / "viggle.py"
_WGP_PATH = _APP / "wgp.py"
_SHARED_UTILS_PATH = _APP / "shared" / "utils" / "utils.py"


def _load_source_functions(path: Path, names: set[str], namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    found = {node.name for node in selected}
    if found != names:
        raise AssertionError(f"Missing source functions from {path}: {sorted(names - found)}")
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def _geometry_functions():
    namespace = {"Image": Image}
    _load_source_functions(
        _PACKING_PATH,
        {"h3_keyframe_fit_crop", "prepare_keyframe_image"},
        namespace,
    )
    _load_source_functions(
        _SHARED_UTILS_PATH,
        {
            "calculate_new_dimensions",
            "rescale_and_crop",
            "calculate_dimensions_and_resize_image",
        },
        namespace,
    )
    return namespace


def _square_marker(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)
    side = min(width, height) // 3
    left = (width - side) // 2
    top = (height - side) // 2
    draw.rectangle((left, top, left + side - 1, top + side - 1), fill="white")
    return image


def _bright_bounds(image: Image.Image) -> tuple[int, int, int, int]:
    xs, ys = [], []
    for y in range(image.height):
        for x in range(image.width):
            if min(image.getpixel((x, y))) > 220:
                xs.append(x)
                ys.append(y)
    if not xs:
        raise AssertionError("Expected the centered white square to remain visible")
    return min(xs), min(ys), max(xs), max(ys)


class H3KeyframeGeometryTests(unittest.TestCase):
    def setUp(self):
        self.geometry = _geometry_functions()

    def _run_shared_h3_anchor_path(
        self,
        source: Image.Image,
        target_size: tuple[int, int],
        *,
        auto_aspect: bool,
    ) -> Image.Image:
        target_height, target_width = target_size
        crop = self.geometry["h3_keyframe_fit_crop"](
            fit_crop=False,
            auto_aspect=auto_aspect,
        )
        fitted, fitted_height, fitted_width = self.geometry[
            "calculate_dimensions_and_resize_image"
        ](
            source,
            target_height,
            target_width,
            1,
            crop,
            block_size=32,
        )
        if not auto_aspect:
            self.assertEqual((fitted_height, fitted_width), target_size)
        prepared = self.geometry["prepare_keyframe_image"](
            fitted,
            target_height,
            target_width,
        )
        self.assertEqual(prepared.size, (target_width, target_height))
        return prepared

    def test_explicit_canvas_cover_crops_landscape_and_portrait_without_distortion(self):
        cases = (
            ((480, 270), (480, 832)),
            ((270, 480), (832, 480)),
        )
        for (source_width, source_height), target_size in cases:
            with self.subTest(source=(source_width, source_height), target=target_size):
                output = self._run_shared_h3_anchor_path(
                    _square_marker(source_width, source_height),
                    target_size,
                    auto_aspect=False,
                )
                left, top, right, bottom = _bright_bounds(output)
                marker_width = right - left + 1
                marker_height = bottom - top + 1
                self.assertLessEqual(abs(marker_width - marker_height), 2)

    def test_auto_canvas_follows_source_aspect_and_keeps_same_aspect_pixels(self):
        # Both dimensions align to H3's 32px grid, so Auto can keep these
        # source aspects exactly through the shared fit helper.
        for width, height in ((480, 256), (256, 480)):
            with self.subTest(source=(width, height)):
                source = _square_marker(width, height)
                output = self._run_shared_h3_anchor_path(
                    source,
                    (height, width),
                    auto_aspect=True,
                )
                self.assertEqual(output.size, source.size)
                self.assertEqual(output.getpixel((width // 2, height // 2)), (255, 255, 255))

    def test_matching_explicit_canvas_does_not_resample_or_shift_content(self):
        source = _square_marker(480, 270)
        prepared = self.geometry["prepare_keyframe_image"](source, 270, 480)
        self.assertIs(prepared, source)

    def test_explicit_aspect_uses_crop_while_auto_keeps_existing_fit_policy(self):
        fit = self.geometry["h3_keyframe_fit_crop"]
        self.assertTrue(fit(fit_crop=False, auto_aspect=False))
        self.assertFalse(fit(fit_crop=False, auto_aspect=True))
        self.assertTrue(fit(fit_crop=True, auto_aspect=True))

    def test_all_h3_keyframe_anchor_paths_use_the_same_non_stretch_helper(self):
        for path, expected_calls in ((_MAIN_PATH, 3), (_VIGGLE_PATH, 1)):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "prepare_keyframe_image"
            ]
            with self.subTest(path=path.name):
                self.assertEqual(len(calls), expected_calls)
                self.assertTrue(all(len(call.args) == 3 for call in calls))
                self.assertTrue(all(not call.keywords for call in calls))

    def test_wgp_applies_h3_fit_policy_to_both_boundary_frames(self):
        source = _WGP_PATH.read_text(encoding="utf-8")
        self.assertIn("h3_keyframe_fit_crop(", source)
        self.assertIn("auto_aspect=_auto_aspect", source)
        self.assertEqual(source.count("_h3_anchor_fit_crop, block_size = block_size"), 2)


if __name__ == "__main__":
    unittest.main()
