"""PTD-only 768px reference detail admission and geometry contracts."""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
import sys
import numpy as np
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


class HTTPError(Exception):
    def __init__(self, *, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ModelContextReached(Exception):
    pass


def _load_prepare_route():
    path = APP / "launch.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item for item in module.body
        if isinstance(item, ast.AsyncFunctionDef)
        and item.name == "_prepare_generation_submission"
    )
    node.decorator_list = []
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            node,
        ],
        type_ignores=[],
    )

    def make_model_def(model_type: str):
        return {
            "architecture": "minimax_h3_ref2va",
            "omni_reference": True,
            "minimax_h3_ptd_refasymopd": model_type.endswith(
                "ptd_refasymopd"
            ),
        }

    namespace = {
        "HTTPException": HTTPError,
        "asyncio": asyncio,
        "_stamp_live_generation_settings_version": lambda _body: None,
        "_is_minimax_h3_identity": lambda *_args: False,
        "_recommended_minimax_h3_encoder": (
            lambda _model_type, _model_def: (_ for _ in ()).throw(
                ModelContextReached()
            )
        ),
        "wgp": types.SimpleNamespace(
            get_model_def=make_model_def,
            get_base_model_type=lambda model_type: model_type,
            apply_model_acceleration_profile=lambda _model_type, _body: {},
        ),
    }
    exec(
        compile(ast.fix_missing_locations(tree), str(path), "exec"),
        namespace,
    )
    return namespace["_prepare_generation_submission"]


def _model_prep_body(model_type: str, detail: str | None):
    body = {
        "model_type": model_type,
        "generation_mode": "video",
        "prompt": "A small tree moves gently in the wind.",
        "minimax_h3_references": [],
    }
    if detail is not None:
        body["minimax_h3_reference_detail"] = detail
    return body


class PTDReferenceDetailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from models.minimax_h3 import ref2va

        cls.ref2va = ref2va
        cls.prepare_route = staticmethod(_load_prepare_route())

    def _run_to_encoder_selection(self, model_type: str, detail: str | None):
        from services import h3_reference_audio

        native_vae_module = types.ModuleType("services.native_vae")
        native_vae_module.native_vae_selection = lambda *_args: None
        duration_module = types.ModuleType("models.minimax_h3.duration")
        duration_module.apply_h3_duration_override = (
            lambda _body, model_def: model_def
        )
        body = _model_prep_body(model_type, detail)
        with (
            mock.patch.object(
                self.ref2va,
                "validate_reference_manifest",
                side_effect=lambda manifest, **_kwargs: list(manifest or []),
            ),
            mock.patch.object(
                h3_reference_audio,
                "measure_video_audio_metadata",
                side_effect=lambda manifest: manifest,
            ),
            mock.patch.dict(
                sys.modules,
                {
                    "services.native_vae": native_vae_module,
                    "models.minimax_h3.duration": duration_module,
                },
            ),
        ):
            try:
                asyncio.run(self.prepare_route(body, prepare_only=True))
            except ModelContextReached as error:
                traceback = error.__traceback__
                while traceback:
                    if traceback.tb_frame.f_code.co_name == "_prepare_generation_submission":
                        return traceback.tb_frame.f_locals["body"]
                    traceback = traceback.tb_next
                raise AssertionError("Model context marker lost route frame") from error

    def test_prepare_route_normalizes_fixed_ptd_detail_before_model_context(self):
        prepared = self._run_to_encoder_selection(
            "minimax_h3_ref2va_ptd_refasymopd",
            "ptd768",
        )
        self.assertEqual(prepared["minimax_h3_reference_detail"], "ptd768")

    def test_prepare_route_keeps_legacy_internal_match_default_for_ptd_fixed(self):
        prepared = self._run_to_encoder_selection(
            "minimax_h3_ref2va_ptd_refasymopd",
            "match",
        )
        self.assertEqual(prepared["minimax_h3_reference_detail"], "ptd768")

    def test_prepare_route_defaults_missing_detail_to_ptd_recipe(self):
        prepared = self._run_to_encoder_selection(
            "minimax_h3_ref2va_ptd_refasymopd",
            None,
        )
        self.assertEqual(prepared["minimax_h3_reference_detail"], "ptd768")

    def test_prepare_route_rejects_ptd_detail_for_ordinary_ref2va(self):
        with self.assertRaises(HTTPError) as raised:
            self._run_to_encoder_selection(
                "minimax_h3_ref2va",
                "ptd768",
            )
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("only available for PTD", raised.exception.detail)

    def test_prepare_route_rejects_nonfixed_detail_for_ptd(self):
        with self.assertRaises(HTTPError) as raised:
            self._run_to_encoder_selection(
                "minimax_h3_ref2va_ptd_refasymopd",
                "max",
            )
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("fixed 'ptd768'", raised.exception.detail)

    def test_ptd_geometry_uses_768_short_edge_for_images_and_videos(self):
        image_size = self.ref2va.resolve_reference_image_size(
            1280,
            720,
            detail="ptd768",
        )
        video_size = self.ref2va.resolve_reference_video_size(
            1272,
            720,
            detail="ptd768",
        )
        self.assertEqual(image_size, (768, 1376))
        self.assertEqual(video_size, (768, 1344))
        self.assertEqual(min(image_size), 768)
        self.assertEqual(min(video_size), 768)
        self.assertTrue(all(size % 32 == 0 for size in image_size + video_size))

    def test_preparation_applies_ptd_geometry_to_the_reference_image(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "reference.png"
            Image.new("RGB", (1280, 720), "navy").save(image_path)
            prepared = self.ref2va.prepare_references(
                [{"type": "image", "path": str(image_path)}],
                num_frames=17,
                target_height=480,
                target_width=864,
                detail="ptd768",
            )
        self.assertEqual(prepared[0].image.size, (1376, 768))

    def test_preparation_applies_ptd_geometry_to_reference_video_frames(self):
        frames = np.zeros((2, 720, 1272, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "reference.mp4"
            video_path.write_bytes(b"synthetic")
            with (
                mock.patch.object(
                    self.ref2va,
                    "extract_reference_video_window",
                    side_effect=lambda path, *_args, **_kwargs: path,
                ),
                mock.patch.object(
                    self.ref2va,
                    "decode_reference_video",
                    return_value=(frames, 24.0, None),
                ),
                mock.patch.object(
                    self.ref2va,
                    "resample_reference_frames",
                    side_effect=lambda decoded, _fps: decoded,
                ),
            ):
                prepared = self.ref2va.prepare_references(
                    [{
                        "type": "video",
                        "path": str(video_path),
                        "effective_duration_seconds": 2.0,
                        "include_audio": False,
                        "has_audio": False,
                    }],
                    num_frames=17,
                    target_height=480,
                    target_width=864,
                    detail="ptd768",
                )
        self.assertEqual(prepared[0].frames.shape[1:3], (768, 1344))

    def test_ptd_size_choice_is_not_accepted_by_generic_api_normalizer(self):
        with self.assertRaisesRegex(ValueError, "only available for PTD"):
            self.ref2va.normalize_reference_detail("ptd768")
        self.assertEqual(
            self.ref2va.normalize_reference_detail("match"),
            "match",
        )
        self.assertEqual(
            self.ref2va.normalize_reference_detail("max"),
            "max",
        )

    def test_ordinary_prepare_route_keeps_match_and_max_choices(self):
        for detail in ("match", "max"):
            with self.subTest(detail=detail):
                prepared = self._run_to_encoder_selection(
                    "minimax_h3_ref2va",
                    detail,
                )
                self.assertEqual(prepared["minimax_h3_reference_detail"], detail)


if __name__ == "__main__":
    unittest.main()
