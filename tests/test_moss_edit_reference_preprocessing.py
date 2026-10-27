"""CPU regression for native MOSS edit references at the shared preprocessing boundary."""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.moss_image.edit_runtime import edit_reference_size, fit_reference_image  # noqa: E402
from models.moss_image.hq_vision import encode_vision_references  # noqa: E402
from models.moss_image.moss_image_handler import family_handler  # noqa: E402


def _load_shared_function(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _resize_shared():
    path = ROOT / "app" / "shared" / "utils" / "utils.py"
    return _load_shared_function(path, "resize_and_remove_background", {
        "Image": Image,
        "np": np,
        "new_session": lambda: object(),
        "remove": lambda image, **_kwargs: image,
        "fit_image_into_canvas": lambda *_args, **_kwargs: (None, None),
    })


def _resolve_fit():
    path = ROOT / "app" / "wgp.py"
    return _load_shared_function(path, "_resolve_image_ref_fit", {})


class _AspectBudgetProcessor:
    size = {"shortest_edge": 56 * 56}
    patch_size = 16

    def __init__(self):
        self.received_sizes = None
        self.budget = None

    def __call__(self, *, images, return_tensors, size):
        assert return_tensors == "pt"
        self.received_sizes = [image.size for image in images]
        self.budget = dict(size)
        grids = []
        for image in images:
            width, height = image.size
            if height / width > 1.5:
                grids.append([1, 40, 18])  # portrait, 184,320 patch pixels
            else:
                grids.append([1, 32, 32])  # near-square, exactly 512x512 patch pixels
        return {
            "pixel_values": torch.zeros((len(images), 1), dtype=torch.float32),
            "image_grid_thw": torch.tensor(grids, dtype=torch.long),
        }


class _BudgetVisionEncoder(nn.Module):
    spatial_merge_size = 2

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, pixel_values, *, grid_thw, return_dict):
        assert return_dict is True
        count = int((grid_thw.prod(-1) // (self.spatial_merge_size ** 2)).sum())
        return SimpleNamespace(pooler_output=torch.zeros((count, 4), device=pixel_values.device))


class MossEditReferencePreprocessingTests(unittest.TestCase):
    def test_editor_definitions_preserve_native_sources_but_keep_downstream_bounds(self):
        text_to_image = family_handler.query_model_def("moss_image", {})
        self.assertFalse(text_to_image.get("preserve_image_ref_native_size", False))

        for variant in ("sft", "hq_v5"):
            with self.subTest(variant=variant):
                definition = family_handler.query_model_def(
                    "moss_image", {"moss_edit": True, "moss_edit_variant": variant}
                )
                self.assertTrue(definition["preserve_image_ref_native_size"])
                self.assertEqual(edit_reference_size(definition), 1024)
                if variant == "hq_v5":
                    self.assertEqual(definition["moss_edit_vision_pixels"], 512 * 512)

    def test_shared_fixed_canvas_preprocessing_preserves_order_and_native_geometry(self):
        definition = family_handler.query_model_def(
            "moss_image", {"moss_edit": True, "moss_edit_variant": "hq_v5"}
        )
        resolve_fit = _resolve_fit()
        resize = _resize_shared()
        fixed_aspect_fit = resolve_fit(definition, auto_aspect=False)
        self.assertEqual(fixed_aspect_fit, 1)

        expected_sizes = [(1320, 2868), (1061, 1057)]
        first = Image.new("RGB", expected_sizes[0], (31, 73, 109))
        second = Image.new("RGB", expected_sizes[1], (181, 127, 43))
        refs = [first, second]
        prepared, masks = resize(
            refs,
            720,
            1280,
            rm_background=False,
            any_background_ref=0,
            fit_into_canvas=fixed_aspect_fit,
            preserve_native_size=definition["preserve_image_ref_native_size"],
        )

        self.assertEqual([image.size for image in prepared], expected_sizes)
        self.assertIs(prepared[0], first)
        self.assertIs(prepared[1], second)
        self.assertEqual([image.getpixel((0, 0)) for image in prepared], [(31, 73, 109), (181, 127, 43)])
        self.assertEqual(masks, [None, None])

        # The selected canvas still governs output; native sources are bounded later by each encoder.
        bounded_vae_refs = [fit_reference_image(image, edit_reference_size(definition)) for image in prepared]
        self.assertTrue(all(max(image.size) <= 1024 for image in bounded_vae_refs))
        self.assertEqual([image.size for image in bounded_vae_refs], [(464, 1024), (1024, 1008)])

        processor = _AspectBudgetProcessor()
        features, grids = encode_vision_references(
            _BudgetVisionEncoder(), processor, prepared,
            max_pixels=definition["moss_edit_vision_pixels"],
        )
        self.assertEqual(processor.received_sizes, expected_sizes)
        self.assertEqual(processor.budget["longest_edge"], 512 * 512)
        self.assertEqual(grids.tolist(), [[1, 40, 18], [1, 32, 32]])
        patch_areas = grids[:, 1] * grids[:, 2] * (16 ** 2)
        self.assertTrue(torch.all(patch_areas <= 512 * 512))
        self.assertEqual([tuple(item.shape) for item in features], [(180, 4), (256, 4)])

        # Without the edit-only flag, the shared fixed-aspect fallback reproduces the bug.
        old_behavior, _ = resize(
            refs,
            720,
            1280,
            rm_background=False,
            any_background_ref=0,
            fit_into_canvas=fixed_aspect_fit,
        )
        self.assertEqual([image.size for image in old_behavior], [(720, 1280), (720, 1280)])
        self.assertNotEqual(old_behavior[0].getpixel((0, 0)), first.getpixel((0, 0)))


if __name__ == "__main__":
    unittest.main()
