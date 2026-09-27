"""CPU tests for Krea Identity Edit reference geometry before grounding and VAE encoding."""
from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image


_ROOT = Path(__file__).resolve().parents[1]
_UTILS_PATH = _ROOT / "app" / "shared" / "utils" / "utils.py"
_KREA_MAIN_PATH = _ROOT / "app" / "models" / "krea2" / "krea2_main.py"


def _load_utils_resize(remove, fit_canvas=None):
    tree = ast.parse(_UTILS_PATH.read_text(encoding="utf-8"), filename=str(_UTILS_PATH))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "resize_and_remove_background"
    )
    namespace = {
        "Image": Image,
        "np": np,
        "new_session": lambda: object(),
        "remove": remove,
        "fit_image_into_canvas": fit_canvas or (lambda *_args, **_kwargs: (None, None)),
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_UTILS_PATH), "exec"), namespace)
    return namespace["resize_and_remove_background"]


def _load_krea_helpers():
    tree = ast.parse(_KREA_MAIN_PATH.read_text(encoding="utf-8"), filename=str(_KREA_MAIN_PATH))
    function_names = {
        "_prepare_grounding_images",
        "_fit_reference_to_target",
        "_reference_position_offset",
        "_fit_all_reference_images",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    namespace = {"Image": Image}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_KREA_MAIN_PATH), "exec"), namespace)

    pipeline_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Krea2Pipeline"
    )
    encode = next(
        node for node in pipeline_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_encode_image_to_latents"
    )
    encode_module = ast.Module(body=[encode], type_ignores=[])
    encode_namespace = {"torch": torch, "Image": Image}
    exec(compile(ast.fix_missing_locations(encode_module), str(_KREA_MAIN_PATH), "exec"), encode_namespace)
    namespace["encode_image_to_latents"] = encode_namespace["_encode_image_to_latents"]
    return namespace


def _convert_image_to_tensor(image):
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1)


class TestKrea2ReferencePreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.krea = _load_krea_helpers()

    def test_native_references_reach_grounding_without_canvas_padding_and_keep_removal(self):
        removed_sizes = []

        def remove_background(image, **_kwargs):
            removed_sizes.append(image.size)
            return image

        resize = _load_utils_resize(remove_background)
        scene = Image.new("RGB", (335, 597), (24, 48, 72))
        subject = Image.new("RGB", (636, 1148), (96, 120, 144))
        prepared, masks = resize(
            [scene, subject],
            1920,
            1088,
            rm_background=True,
            any_background_ref=1,
            fit_into_canvas=1,
            preserve_native_size=True,
        )

        self.assertEqual([image.size for image in prepared], [(335, 597), (636, 1148)])
        self.assertEqual(masks, [None, None])
        self.assertEqual(removed_sizes, [(636, 1148)])  # The scene/background ref keeps its existing exemption.
        self.assertEqual(prepared[0].getpixel((0, 0)), scene.getpixel((0, 0)))
        self.assertEqual(prepared[1].getpixel((0, 0)), subject.getpixel((0, 0)))

        grounding = self.krea["_prepare_grounding_images"](prepared, 768)
        self.assertEqual([image.size for image in grounding], [(335, 597), (425, 768)])
        self.assertEqual(grounding[0].getpixel((0, 0)), scene.getpixel((0, 0)))

    def test_default_reference_preparation_still_fits_other_models_to_canvas(self):
        resize = _load_utils_resize(lambda image, **_kwargs: image)
        images = [Image.new("RGB", (335, 597)), Image.new("RGB", (636, 1148))]
        prepared, _ = resize(
            images,
            1920,
            1088,
            rm_background=False,
            any_background_ref=1,
            fit_into_canvas=1,
        )
        self.assertEqual([image.size for image in prepared], [(1920, 1088), (1920, 1088)])

    def test_outpaint_background_reference_still_uses_canvas_mapping(self):
        outpaint_calls = []
        output_mask = object()

        def fit_canvas(image, size, _inpaint_color=127.5, **kwargs):
            outpaint_calls.append((image.size, size, kwargs))
            return Image.new("RGB", (size[1], size[0]), (127, 127, 127)), output_mask

        resize = _load_utils_resize(lambda image, **_kwargs: image, fit_canvas)
        image = Image.new("RGB", (335, 597))
        prepared, masks = resize(
            [image],
            1920,
            1088,
            rm_background=False,
            any_background_ref=1,
            fit_into_canvas=1,
            outpainting_dims=(0, 0, 16, 32),
            background_ref_outpainted=True,
            preserve_native_size=True,
        )
        self.assertEqual(outpaint_calls[0][0:2], (image.size, (1088, 1920)))
        self.assertEqual(prepared[0].size, (1920, 1088))
        self.assertIs(masks[0], output_mask)

    def test_native_geometry_produces_centered_vae_grid_and_preserves_outpaint_offset(self):
        scene = Image.new("RGB", (335, 597), (20, 30, 40))
        fit_all = self.krea["_fit_all_reference_images"](True, "KI")
        self.assertTrue(fit_all)
        fit, resize_to_target = self.krea["_fit_reference_to_target"](None, fit_all, 1)
        self.assertEqual((fit, resize_to_target), (True, True))

        class FakeVae:
            dtype = torch.float32
            config = types.SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16)

            def __init__(self):
                self.input_size = None

            def encode(self, tensor):
                self.input_size = tuple(tensor.shape[-2:])
                latent_shape = (tensor.shape[-2] // 8, tensor.shape[-1] // 8)
                latents = torch.zeros((1, 16, 1, *latent_shape), dtype=tensor.dtype)
                return types.SimpleNamespace(
                    latent_dist=types.SimpleNamespace(mode=lambda: latents),
                )

        vae = FakeVae()
        pipeline = types.SimpleNamespace(
            vae=vae,
            channels=16,
            compression=8,
            transformer=types.SimpleNamespace(config=types.SimpleNamespace(patch=2)),
        )
        shared_module = types.ModuleType("shared")
        shared_utils_module = types.ModuleType("shared.utils")
        shared_utils_impl = types.ModuleType("shared.utils.utils")
        shared_utils_impl.convert_image_to_tensor = _convert_image_to_tensor
        with mock.patch.dict(sys.modules, {
            "shared": shared_module,
            "shared.utils": shared_utils_module,
            "shared.utils.utils": shared_utils_impl,
        }):
            latents = self.krea["encode_image_to_latents"](
                pipeline,
                scene,
                1920,
                1088,
                "cpu",
                torch.float32,
                fit=fit,
                resize_to_target=resize_to_target,
            )

        self.assertEqual(vae.input_size, (1088, 608))
        grid_h, grid_w = latents.shape[-2] // 2, latents.shape[-1] // 2
        self.assertEqual((grid_h, grid_w), (68, 38))
        self.assertEqual(
            self.krea["_reference_position_offset"](1088 // 16, 1920 // 16, grid_h, grid_w, None),
            (0, 41),
        )
        outpaint_offset = (8, 16)
        self.assertEqual(self.krea["_fit_reference_to_target"](outpaint_offset, True, 1), (False, False))
        self.assertEqual(
            self.krea["_reference_position_offset"](68, 120, grid_h, grid_w, outpaint_offset),
            outpaint_offset,
        )
        self.assertFalse(self.krea["_fit_all_reference_images"](False, "KI"))
        self.assertTrue(self.krea["_fit_all_reference_images"](False, "I"))

    def test_grid_alignment_crops_native_reference_before_resize(self):
        pixels = np.full((1148, 636, 3), 100, dtype=np.uint8)
        pixels[:, :5] = (255, 0, 0)
        pixels[:, -5:] = (0, 255, 0)
        subject = Image.fromarray(pixels)
        captured = []

        def capture_conversion(image):
            captured.append(image.copy())
            return _convert_image_to_tensor(image)

        class FakeVae:
            dtype = torch.float32
            config = types.SimpleNamespace(latents_mean=[0.0] * 16, latents_std=[1.0] * 16)

            def encode(self, tensor):
                latent_shape = (tensor.shape[-2] // 8, tensor.shape[-1] // 8)
                latents = torch.zeros((1, 16, 1, *latent_shape), dtype=tensor.dtype)
                return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: latents))

        pipeline = types.SimpleNamespace(
            vae=FakeVae(),
            channels=16,
            compression=8,
            transformer=types.SimpleNamespace(config=types.SimpleNamespace(patch=2)),
        )
        shared_module = types.ModuleType("shared")
        shared_utils_module = types.ModuleType("shared.utils")
        shared_utils_impl = types.ModuleType("shared.utils.utils")
        shared_utils_impl.convert_image_to_tensor = capture_conversion
        with mock.patch.dict(sys.modules, {
            "shared": shared_module,
            "shared.utils": shared_utils_module,
            "shared.utils.utils": shared_utils_impl,
        }):
            self.krea["encode_image_to_latents"](
                pipeline,
                subject,
                1920,
                1088,
                "cpu",
                torch.float32,
                fit=True,
                resize_to_target=True,
            )

        self.assertEqual(captured[0].size, (592, 1088))
        encoded = np.asarray(captured[0])
        np.testing.assert_allclose(encoded[:, 0, :].mean(axis=0), (100, 100, 100), atol=20)
        np.testing.assert_allclose(encoded[:, -1, :].mean(axis=0), (100, 100, 100), atol=20)


if __name__ == "__main__":
    unittest.main()
