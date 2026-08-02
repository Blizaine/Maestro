"""Model-free regressions for the first GitHub issue stability batch."""
from __future__ import annotations

import ast
import math
import os
import sys
import types
import unittest
from unittest import mock


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LAUNCH_PATH = os.path.join(_ROOT, "app", "launch.py")
_WGP_PATH = os.path.join(_ROOT, "app", "wgp.py")
_LTX2_PATH = os.path.join(_ROOT, "app", "models", "ltx2", "ltx2.py")
_LTX2_INPAINT_PATH = os.path.join(
    _ROOT,
    "app",
    "models",
    "ltx2",
    "inpainting.py",
)
_LTX2_DISTILLED_PATH = os.path.join(
    _ROOT,
    "app",
    "models",
    "ltx2",
    "ltx_pipelines",
    "distilled.py",
)
_LTX2_HELPERS_PATH = os.path.join(
    _ROOT,
    "app",
    "models",
    "ltx2",
    "ltx_pipelines",
    "utils",
    "helpers.py",
)
_CLIENT_PATH = os.path.join(_ROOT, "ui", "src", "api", "client.ts")
_STORE_PATH = os.path.join(_ROOT, "ui", "src", "stores", "useStore.ts")
_INPUTS_PATH = os.path.join(
    _ROOT, "ui", "src", "components", "Sidebar", "InputsPanel.tsx",
)
_OUTPAINT_CONTROLS_PATH = os.path.join(
    _ROOT,
    "ui",
    "src",
    "components",
    "Sidebar",
    "OutpaintControls.tsx",
)
_OUTPAINT_CANVAS_PATH = os.path.join(
    _ROOT,
    "ui",
    "src",
    "components",
    "Sidebar",
    "OutpaintCanvas.tsx",
)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_functions(path: str, names: tuple[str, ...], namespace=None) -> dict:
    source = _read(path)
    tree = ast.parse(source, filename=os.path.relpath(path, _ROOT))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(selected) != len(names):
        found = {node.name for node in selected}
        raise AssertionError(f"Missing functions: {set(names) - found}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    loaded = dict(namespace or {})
    exec(compile(module, os.path.relpath(path, _ROOT), "exec"), loaded)
    return loaded


class TestOutpaintSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolve = staticmethod(
            _load_functions(
                _LAUNCH_PATH,
                ("_resolve_outpaint_sampling",),
            )["_resolve_outpaint_sampling"]
        )

    def test_ltxv_uses_its_model_default_instead_of_eight(self):
        self.assertEqual(
            self.resolve({}, "ltxv_13B", {}, {
                "num_inference_steps": 30,
                "guidance_scale": 3.0,
            }),
            (30, 3.0),
        )

    def test_explicit_outpaint_sampling_is_preserved(self):
        self.assertEqual(
            self.resolve(
                {"num_inference_steps": 24, "guidance_scale": 2.5},
                "ltxv_13B",
                {},
                {"num_inference_steps": 30},
            ),
            (24, 2.5),
        )

    def test_ltxv_rejects_an_explicit_invalid_step_count(self):
        with self.assertRaisesRegex(ValueError, "at least 20"):
            self.resolve(
                {"num_inference_steps": 8},
                "ltxv_13B",
                {},
                {"num_inference_steps": 30},
            )

    def test_locked_distilled_schedule_uses_model_default(self):
        self.assertEqual(
            self.resolve(
                {"num_inference_steps": 30, "guidance_scale": 7},
                "ltx2_22B",
                {
                    "lock_inference_steps": True,
                    "lock_guidance_scale": True,
                },
                {"num_inference_steps": 8, "guidance_scale": 1},
            ),
            (8, 1.0),
        )

    def test_frontend_sends_the_visible_advanced_values(self):
        client = _read(_CLIENT_PATH)
        store = _read(_STORE_PATH)
        for field in (
            "num_inference_steps?: number",
            "guidance_scale?: number",
            "negative_prompt?: string",
        ):
            self.assertIn(field, client)
        self.assertIn(
            "num_inference_steps: "
            "(state.params.num_inference_steps as number)",
            store,
        )
        self.assertIn(
            "guidance_scale: (state.params.guidance_scale as number)",
            store,
        )


class TestOutpaintTimingAndWindowUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolve_timing = staticmethod(
            _load_functions(
                _LAUNCH_PATH,
                ("_resolve_outpaint_video_timing",),
                {"math": math},
            )["_resolve_outpaint_video_timing"]
        )

    def test_thirty_fps_source_uses_diffusers_reference_rate(self):
        duration, target_fps, target_frames = self.resolve_timing(
            585,
            30,
            {
                "fps": 25,
                "frames_minimum": 17,
                "frames_steps": 8,
            },
            reference_fps=24,
        )
        self.assertAlmostEqual(duration, 19.5)
        self.assertEqual(target_fps, 24)
        self.assertEqual(target_frames, 473)

    def test_reference_rate_keeps_source_tail_on_complete_latent_grid(self):
        duration, target_fps, target_frames = self.resolve_timing(
            309,
            30,
            {
                "fps": 25,
                "frames_minimum": 17,
                "frames_steps": 8,
            },
            reference_fps=24,
        )
        self.assertAlmostEqual(duration, 10.3)
        self.assertEqual(target_fps, 24)
        self.assertEqual(target_frames, 249)

    def test_production_timing_retains_native_rate_and_frame_grid(self):
        duration, target_fps, target_frames = self.resolve_timing(
            309,
            30,
            {
                "fps": 25,
                "frames_minimum": 17,
                "frames_steps": 8,
            },
        )
        self.assertAlmostEqual(duration, 10.3)
        self.assertEqual(target_fps, 30)
        self.assertEqual(target_frames, 305)

    def test_outpaint_tracks_trim_duration_and_hides_generic_resolution(self):
        controls = _read(
            os.path.join(
                _ROOT,
                "ui",
                "src",
                "components",
                "Sidebar",
                "OutpaintControls.tsx",
            )
        )
        duration = _read(
            os.path.join(
                _ROOT,
                "ui",
                "src",
                "components",
                "Sidebar",
                "DurationSlider.tsx",
            )
        )
        advanced = _read(
            os.path.join(
                _ROOT,
                "ui",
                "src",
                "components",
                "Sidebar",
                "AdvancedSettings.tsx",
            )
        )
        self.assertIn("Math.ceil(selectedDuration) + 1", controls)
        self.assertIn("setWindowSize(nextWindow)", controls)
        self.assertIn(
            "const duration = isOutpaint "
            "? trimmedOutpaintDuration : studioDuration",
            duration,
        )
        self.assertIn(
            "!isOutpaint && !modelOptions?.hide_resolution_presets",
            advanced,
        )


class TestOutpaintCurrentModelStack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_defs = {
            "ltx2_22B_distilled_1_1": {
                "architecture": "ltx2_22B",
                "ltx2_pipeline": "distilled",
            },
            "ltx2_22B_distilled_fp8": {
                "architecture": "ltx2_22B",
                "ltx2_pipeline": "distilled",
            },
            "ltx2_22B_1_1": {"architecture": "ltx2_22B"},
            "ltx2_22B_fp8": {"architecture": "ltx2_22B"},
            "ltx2_22B": {"architecture": "ltx2_22B"},
        }
        model_urls = {
            "ltx2_22B_distilled_1_1": [
                "https://example.test/ltx-2.3-22b-distilled-1.1.safetensors",
            ],
            "ltx2_22B_distilled_fp8": [
                "https://example.test/ltx-2.3-22b-distilled-fp8.safetensors",
            ],
            "ltx2_22B_1_1": [
                "https://example.test/ltx-2.3-22b-dev.safetensors",
            ],
            "ltx2_22B_fp8": [
                "https://example.test/ltx-2.3-22b-dev-fp8.safetensors",
            ],
            "ltx2_22B": [
                "https://example.test/ltx-2.3-22b-base.safetensors",
            ],
        }
        fake_wgp = types.SimpleNamespace(
            get_model_def=lambda model_type: model_defs.get(model_type),
            get_model_recursive_prop=lambda model_type, prop, **_: (
                model_urls.get(model_type, []) if prop == "URLs" else None
            ),
        )
        cls.resolve_model = staticmethod(
            _load_functions(
                _LAUNCH_PATH,
                ("_resolve_ltx2_outpaint_reference_model",),
                {"os": os, "wgp": fake_wgp},
            )["_resolve_ltx2_outpaint_reference_model"]
        )

    def _files_locator_modules(self, installed_filename):
        files_locator = types.ModuleType("shared.utils.files_locator")
        files_locator.locate_file = lambda filename, **_: (
            os.path.join("ckpts", filename)
            if filename == installed_filename
            else None
        )
        shared_utils = types.ModuleType("shared.utils")
        shared_utils.files_locator = files_locator
        shared = types.ModuleType("shared")
        shared.utils = shared_utils
        return {
            "shared": shared,
            "shared.utils": shared_utils,
            "shared.utils.files_locator": files_locator,
        }

    def test_requested_dev_model_is_retained(self):
        modules = self._files_locator_modules("")
        with mock.patch.dict(sys.modules, modules):
            self.assertEqual(
                self.resolve_model("ltx2_22B_1_1"),
                "ltx2_22B_1_1",
            )

    def test_distilled_request_routes_to_installed_dev_fp8(self):
        modules = self._files_locator_modules(
            "ltx-2.3-22b-dev-fp8.safetensors",
        )
        with mock.patch.dict(sys.modules, modules):
            self.assertEqual(
                self.resolve_model("ltx2_22B_distilled_1_1"),
                "ltx2_22B_fp8",
            )

    def test_distilled_request_is_retained_without_installed_dev(self):
        modules = self._files_locator_modules("")
        with mock.patch.dict(sys.modules, modules):
            self.assertEqual(
                self.resolve_model("ltx2_22B_distilled_1_1"),
                "ltx2_22B_distilled_1_1",
            )

    def test_outpaint_endpoint_records_requested_model_before_routing(self):
        launch = _read(_LAUNCH_PATH)
        outpaint = launch.split(
            '@api.post("/api/v1/outpaint")',
            1,
        )[1]
        self.assertLess(
            outpaint.index("requested_model_type = model_type"),
            outpaint.index(
                "model_type = "
                "_resolve_ltx2_outpaint_reference_model(model_type)"
            ),
        )


class TestMaskPreservingOutpaint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app_root = os.path.join(_ROOT, "app")
        if app_root not in sys.path:
            sys.path.insert(0, app_root)
        cls.resolve_geometry = staticmethod(
            _load_functions(
                _LAUNCH_PATH,
                ("_resolve_outpaint_canvas_geometry",),
                {
                    "math": math,
                    "_OUTPAINT_PIXEL_BUDGETS": {
                        "480p": 480 * 848,
                        "540p": 540 * 960,
                        "720p": 720 * 1280,
                        "1080p": 1088 * 1920,
                    },
                },
            )["_resolve_outpaint_canvas_geometry"]
        )
        cls.cap_single_stage_window = staticmethod(
            _load_functions(
                _LAUNCH_PATH,
                ("_cap_outpaint_single_stage_window",),
                {
                    "_OUTPAINT_SINGLE_STAGE_REFERENCE_PIXELS": 704 * 1280,
                    "_OUTPAINT_SINGLE_STAGE_REFERENCE_FRAMES": 257,
                },
            )["_cap_outpaint_single_stage_window"]
        )
        cls.parse_dims = staticmethod(
            _load_functions(
                _WGP_PATH,
                ("get_outpainting_dims",),
            )["get_outpainting_dims"]
        )

    def test_portrait_canvas_is_aligned_before_mask_geometry(self):
        geometry = self.resolve_geometry(
            896,
            512,
            540,
            541,
            0,
            0,
            "auto",
            64,
        )
        self.assertEqual(
            (geometry["final_w"], geometry["final_h"]),
            (896, 1600),
        )
        self.assertEqual(
            (geometry["overlay_w"], geometry["overlay_h"]),
            (896, 512),
        )
        self.assertEqual(geometry["overlay_x"], 0)
        self.assertIn(geometry["overlay_y"], (543, 544))

    def test_fractional_geometry_survives_wgp_parser(self):
        self.assertEqual(
            self.parse_dims("106.25 106.25 0 0"),
            [106.25, 106.25, 0.0, 0.0],
        )

    def test_native_portrait_canvas_uses_a_vram_bounded_window(self):
        self.assertEqual(
            self.cap_single_stage_window(501, 896, 1600, 17, 501),
            161,
        )
        self.assertEqual(
            self.cap_single_stage_window(501, 704, 1280, 17, 501),
            501,
        )

    def test_mask_and_neutral_canvas_protect_the_source_rect(self):
        import torch
        from models.ltx2.inpainting import (
            _build_outpainting_mask_cthw,
            _paint_ltx2_masked_control_video,
        )

        source = torch.zeros((3, 1, 64, 128), dtype=torch.float32)
        mask = _build_outpainting_mask_cthw(
            source,
            [0, 0, 100, 0],
        )
        self.assertIsNotNone(mask)
        self.assertTrue(torch.all(mask[:, :, :, :64] == 1))
        self.assertTrue(torch.all(mask[:, :, :, 64:] == 0))

        prepared = _paint_ltx2_masked_control_video(source, mask)
        expected_gray = torch.tensor(
            [128, 128, 128],
            dtype=torch.float32,
        ).div(127.5).sub(1.0)
        self.assertTrue(
            torch.allclose(
                prepared[:, 0, 0, 0],
                expected_gray,
            )
        )
        self.assertTrue(torch.all(prepared[:, :, :, 64:] == 0))

    def test_official_outpaint_green_marker_remains_exact(self):
        import torch
        from models.ltx2.inpainting import (
            _build_outpainting_mask_cthw,
            _paint_ltx2_inpaint_control_video,
        )

        source = torch.zeros((3, 1, 64, 128), dtype=torch.float32)
        mask = _build_outpainting_mask_cthw(source, [0, 0, 100, 0])
        prepared = _paint_ltx2_inpaint_control_video(source, mask)
        expected_green = torch.tensor(
            [102, 255, 0],
            dtype=torch.float32,
        ).div(127.5).sub(1.0)
        self.assertTrue(
            torch.allclose(prepared[:, 0, 0, 0], expected_green)
        )
        self.assertTrue(torch.all(prepared[:, :, :, 64:] == 0))

    def test_source_attention_uses_area_and_causal_temporal_reduction(self):
        import torch
        from models.ltx2.ltx_pipelines.utils.helpers import (
            _outpaint_source_attention_to_latents,
        )

        # 9 pixel frames map to two causal latent frames: frame zero remains
        # independent and frames 1..8 are averaged into latent frame one.
        generation = torch.ones((1, 1, 9, 4, 8))
        generation[:, :, 0, :, 4:] = 0
        generation[:, :, 1:, :, 2:] = 0
        attention = _outpaint_source_attention_to_latents(
            generation,
            target_frames=2,
            target_h=2,
            target_w=4,
        )

        self.assertEqual(tuple(attention.shape), (1, 1, 2, 2, 4))
        self.assertTrue(torch.all(attention[:, :, 0, :, :2] == 0))
        self.assertTrue(torch.all(attention[:, :, 0, :, 2:] == 1))
        self.assertTrue(torch.all(attention[:, :, 1, :, :1] == 0))
        self.assertTrue(torch.all(attention[:, :, 1, :, 1:] == 1))

    def test_source_attention_retains_partial_outpaint_boundary_rows(self):
        import torch
        from models.ltx2.ltx_pipelines.utils.helpers import (
            _outpaint_source_attention_to_latents,
        )

        # This mirrors the 448x832 portrait geometry from the reported
        # single-shot regression. The protected 254px source rectangle lands
        # across three full latent rows plus two 48.4%-covered boundary rows.
        # Thresholding at 0.5 would incorrectly discard both boundary rows.
        generation = torch.ones((1, 1, 1, 832, 448))
        generation[:, :, :, 289:543, :] = 0
        attention = _outpaint_source_attention_to_latents(
            generation,
            target_frames=1,
            target_h=13,
            target_w=7,
        )

        row_weights = attention[0, 0, 0, :, 0]
        self.assertTrue(
            torch.allclose(
                row_weights,
                torch.tensor(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.484375,
                        1.0,
                        1.0,
                        1.0,
                        0.484375,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            )
        )
        self.assertAlmostEqual(float(attention.sum()), 27.78125, places=5)
        self.assertEqual(int((attention > 0).sum()), 35)

    def test_single_stage_edge_extension_cannot_carry_green_canvas(self):
        import torch
        from models.ltx2.inpainting import (
            _edge_extend_ltx2_masked_control_video,
        )

        source = torch.full((3, 1, 64, 128), -1.0)
        mask = torch.ones((1, 1, 64, 128), dtype=torch.uint8)
        mask[:, :, 16:48, 32:96] = 0
        protected = torch.linspace(
            -0.8,
            0.8,
            3 * 32 * 64,
        ).reshape(3, 1, 32, 64)
        source[:, :, 16:48, 32:96] = protected

        prepared = _edge_extend_ltx2_masked_control_video(source, mask)

        self.assertTrue(
            torch.equal(
                prepared[:, :, 16:48, 32:96],
                protected,
            )
        )
        self.assertTrue(
            torch.equal(
                prepared[:, :, 0, 0],
                protected[:, :, 0, 0],
            )
        )
        expected_green = torch.tensor(
            [102, 255, 0],
            dtype=torch.float32,
        ).div(127.5).sub(1.0)
        self.assertFalse(
            torch.allclose(prepared[:, 0, 0, 0], expected_green)
        )

    def test_masked_reference_retains_canvas_and_builds_official_attention(self):
        import torch
        from models.ltx2.ltx_core.components.patchifiers import (
            VideoLatentPatchifier,
        )
        from models.ltx2.ltx_core.conditioning.types.reference_video_cond import (
            VideoConditionByReferenceLatent,
        )
        from models.ltx2.ltx_core.types import (
            LatentState,
            SpatioTemporalScaleFactors,
        )

        patchifier = VideoLatentPatchifier(1)
        tools = types.SimpleNamespace(
            patchifier=patchifier,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
            fps=25.0,
        )
        latent = torch.arange(8, dtype=torch.float32).reshape(
            1, 1, 1, 2, 4
        )
        cross_mask = torch.zeros_like(latent)
        cross_mask[:, :, :, :, 2] = 0.25
        cross_mask[:, :, :, :, 3] = 0.75
        state = LatentState(
            latent=torch.zeros((1, 8, 1)),
            denoise_mask=torch.ones((1, 8, 1)),
            positions=torch.zeros((1, 3, 8, 2)),
            clean_latent=torch.zeros((1, 8, 1)),
        )

        result = VideoConditionByReferenceLatent(
            latent=latent,
            reference_cross_mask=cross_mask,
        ).apply_to(state, tools)

        self.assertEqual(result.latent.shape[1], 16)
        self.assertEqual(
            result.latent[:, 8:, 0].tolist(),
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
        )
        self.assertTrue(torch.is_floating_point(result.attention_mask))
        self.assertEqual(tuple(result.attention_mask.shape), (1, 16, 16))
        expected_cross = torch.tensor(
            [0.0, 0.0, 0.25, 0.75, 0.0, 0.0, 0.25, 0.75]
        )
        self.assertTrue(
            torch.allclose(result.attention_mask[0, 0, 8:], expected_cross)
        )
        self.assertTrue(
            torch.allclose(result.attention_mask[0, 8:, 0], expected_cross)
        )
        self.assertTrue(torch.all(result.attention_mask[:, :8, :8]))
        self.assertTrue(torch.all(result.attention_mask[:, 8:, 8:]))

    def test_unpruned_reference_matches_stable_keyframe_layout(self):
        import torch
        from models.ltx2.ltx_core.components.patchifiers import (
            VideoLatentPatchifier,
        )
        from models.ltx2.ltx_core.conditioning.types.keyframe_cond import (
            VideoConditionByKeyframeIndex,
        )
        from models.ltx2.ltx_core.conditioning.types.reference_video_cond import (
            VideoConditionByReferenceLatent,
        )
        from models.ltx2.ltx_core.types import (
            LatentState,
            SpatioTemporalScaleFactors,
        )

        patchifier = VideoLatentPatchifier(1)
        tools = types.SimpleNamespace(
            patchifier=patchifier,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
            fps=25.0,
        )
        latent = torch.arange(8, dtype=torch.float32).reshape(
            1, 1, 1, 2, 4
        )
        state = LatentState(
            latent=torch.zeros((1, 8, 1)),
            denoise_mask=torch.ones((1, 8, 1)),
            positions=torch.zeros((1, 3, 8, 2)),
            clean_latent=torch.zeros((1, 8, 1)),
        )

        keyframe = VideoConditionByKeyframeIndex(
            keyframes=latent,
            frame_idx=0,
            strength=1.0,
        ).apply_to(state, tools)
        reference = VideoConditionByReferenceLatent(
            latent=latent,
            frame_idx=0,
            strength=1.0,
        ).apply_to(state, tools)

        self.assertTrue(torch.equal(reference.latent, keyframe.latent))
        self.assertTrue(
            torch.equal(reference.denoise_mask, keyframe.denoise_mask)
        )
        self.assertTrue(
            torch.equal(reference.positions, keyframe.positions)
        )
        self.assertTrue(
            torch.equal(reference.clean_latent, keyframe.clean_latent)
        )

    def test_boolean_self_attention_mask_stays_compact_for_sdpa(self):
        import torch
        from models.ltx2.ltx_core.model.transformer.transformer_args import (
            TransformerArgsPreprocessor,
        )

        preprocessor = object.__new__(TransformerArgsPreprocessor)
        mask = torch.tensor(
            [[[True, False], [True, True]]],
            dtype=torch.bool,
        )

        prepared = preprocessor._prepare_self_attention_mask(
            mask,
            torch.bfloat16,
        )

        self.assertEqual(prepared.dtype, torch.bool)
        self.assertEqual(tuple(prepared.shape), (1, 2, 1, 2))
        query = torch.randn((1, 2, 1, 4), dtype=torch.float32)
        output = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            query.transpose(1, 2),
            query.transpose(1, 2),
            attn_mask=prepared.transpose(1, 2),
        )
        self.assertEqual(tuple(output.shape), (1, 1, 2, 4))

    def test_clear_conditioning_drops_reference_attention_mask(self):
        import torch
        from models.ltx2.ltx_core.components.patchifiers import (
            VideoLatentPatchifier,
        )
        from models.ltx2.ltx_core.tools import VideoLatentTools
        from models.ltx2.ltx_core.types import LatentState, VideoLatentShape

        tools = VideoLatentTools(
            patchifier=VideoLatentPatchifier(1),
            target_shape=VideoLatentShape(1, 1, 1, 2, 4),
            fps=25.0,
        )
        state = LatentState(
            latent=torch.zeros((1, 16, 1)),
            denoise_mask=torch.ones((1, 16, 1)),
            positions=torch.zeros((1, 3, 16, 2)),
            clean_latent=torch.zeros((1, 16, 1)),
            attention_mask=torch.ones((1, 16, 16), dtype=torch.bool),
        )

        cleared = tools.clear_conditioning(state)

        self.assertEqual(tuple(cleared.latent.shape), (1, 8, 1))
        self.assertIsNone(cleared.attention_mask)

    def test_boundary_blend_cannot_reach_deep_into_protected_source(self):
        import torch
        from models.ltx2.inpainting import _apply_ltx2_mask_blend

        height, width = 96, 128
        generated = torch.ones(
            (3, 1, height, width),
            dtype=torch.float32,
        )
        source = torch.zeros_like(generated)
        mask = torch.ones(
            (1, 1, height, width),
            dtype=torch.uint8,
        )
        mask[:, :, 24:72, 16:112] = 0

        result = _apply_ltx2_mask_blend(
            generated,
            source,
            mask,
            1,
            height,
            width,
            mask_low_res_dilation=6,
            source_feather_pixels=8,
        )

        # The generated canvas is never softened by the source-restoration
        # pass, while the source interior becomes exactly source-valued.
        self.assertTrue(torch.equal(result[:, :, :16], generated[:, :, :16]))
        self.assertTrue(
            torch.equal(
                result[:, :, 32:64, 24:104],
                source[:, :, 32:64, 24:104],
            )
        )
        # A small non-zero band remains for hiding the rectangle seam.
        self.assertTrue(torch.all(result[:, :, 24, 24:104] > 0))
        self.assertTrue(torch.all(result[:, :, 31, 24:104] > 0))

    def test_official_gaussian_restore_never_grades_generated_canvas(self):
        import torch
        from models.ltx2.inpainting import _apply_ltx2_mask_blend

        height, width = 160, 128
        generated = torch.zeros((3, 1, height, width), dtype=torch.float32)
        generated[0].fill_(0.2)
        generated[1].fill_(0.7)
        generated[2].fill_(0.4)
        source = torch.zeros_like(generated)
        mask = torch.ones((1, 1, height, width), dtype=torch.uint8)
        mask[:, :, 32:128, 8:120] = 0

        result = _apply_ltx2_mask_blend(
            generated,
            source,
            mask,
            1,
            height,
            width,
            source_feather_pixels=8,
            match_generated_canvas=True,
            blend_mode="gaussian",
        )

        self.assertTrue(torch.equal(result[:, :, :32], generated[:, :, :32]))
        self.assertTrue(torch.equal(result[:, :, 64:96, 32:96], source[:, :, 64:96, 32:96]))
        self.assertTrue(torch.all(result[:, :, 32, 32:96] > 0))

    def test_green_mask_is_sanitized_before_bounded_laplacian_blend(self):
        import torch
        from models.ltx2.inpainting import (
            _apply_ltx2_mask_blend,
            _paint_ltx2_masked_control_video,
        )

        height, width = 96, 128
        clean_source = torch.zeros(
            (3, 2, height, width),
            dtype=torch.float32,
        )
        generated = torch.zeros_like(clean_source)
        mask = torch.ones(
            (1, 2, height, width),
            dtype=torch.uint8,
        )
        mask[:, :, 24:72, 8:120] = 0
        masked_source = _paint_ltx2_masked_control_video(
            clean_source,
            mask,
        )

        result = _apply_ltx2_mask_blend(
            generated,
            masked_source,
            mask,
            2,
            height,
            width,
            mask_low_res_dilation=6,
            source_feather_pixels=8,
            match_generated_canvas=False,
            full_frame_laplacian=False,
        )

        self.assertTrue(
            torch.allclose(
                result,
                clean_source,
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_refinement_mask_keeps_odd_boundary_as_fractional_area(self):
        import torch
        from models.ltx2.ltx_pipelines.distilled import (
            _coerce_refinement_mask_cthw,
        )

        mask = torch.ones((1, 1, 8, 4), dtype=torch.uint8)
        mask[:, :, 3:5, :] = 0

        resized = _coerce_refinement_mask_cthw(
            mask,
            height=4,
            width=2,
            num_frames=1,
        )

        self.assertEqual(resized.dtype, torch.float32)
        self.assertTrue(torch.all(resized[:, :, 0] == 1.0))
        self.assertTrue(torch.all(resized[:, :, 1] == 0.5))
        self.assertTrue(torch.all(resized[:, :, 2] == 0.5))
        self.assertTrue(torch.all(resized[:, :, 3] == 1.0))

    def test_refinement_source_removes_marker_before_area_resize(self):
        import torch
        from models.ltx2.inpainting import (
            _paint_ltx2_inpaint_control_video,
        )
        from models.ltx2.ltx_pipelines.distilled import (
            _coerce_refinement_source_cthw,
        )

        clean = torch.zeros((3, 1, 8, 4), dtype=torch.float32)
        mask = torch.ones((1, 1, 8, 4), dtype=torch.uint8)
        mask[:, :, 3:5, :] = 0
        marked = _paint_ltx2_inpaint_control_video(clean, mask)

        resized = _coerce_refinement_source_cthw(
            [(marked, 0, 1.0)],
            height=4,
            width=2,
            num_frames=1,
            generation_mask=mask,
        )

        self.assertTrue(
            torch.allclose(
                resized,
                torch.zeros_like(resized),
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_refinement_handoff_decodes_blends_then_reencodes_pixels(self):
        from collections.abc import Callable

        import torch

        calls = []
        decoded = torch.full(
            (2, 4, 6, 3),
            255,
            dtype=torch.uint8,
        )
        expected_latent = torch.ones((1, 2, 1, 1, 1))
        captured = {}

        def fake_decode(*args, **kwargs):
            calls.append("decode")
            self.assertEqual(kwargs["expected_frames"], 2)
            self.assertEqual(kwargs["expected_height"], 4)
            self.assertEqual(kwargs["expected_width"], 6)
            return decoded

        def fake_blend(video, source, mask, *args, **kwargs):
            calls.append("blend")
            self.assertEqual(tuple(video.shape), (3, 2, 4, 6))
            self.assertEqual(kwargs["mask_low_res_dilation"], 5)
            self.assertFalse(kwargs["match_generated_canvas"])
            self.assertTrue(kwargs["full_frame_laplacian"])
            return torch.full_like(video, 127)

        def fake_encode(video, *args, **kwargs):
            calls.append("encode")
            captured["video"] = video.detach().clone()
            return expected_latent

        handoff_functions = _load_functions(
            _LTX2_DISTILLED_PATH,
            (
                "_resize_refinement_video_cthw",
                "_decode_blend_reencode_outpaint",
            ),
            {
                "Callable": Callable,
                "F": torch.nn.functional,
                "TilingConfig": object,
                "torch": torch,
                "_apply_ltx2_mask_blend": fake_blend,
                "vae_decode_video_to_tensor": fake_decode,
                "vae_encode_video": fake_encode,
            },
        )
        handoff = handoff_functions[
            "_decode_blend_reencode_outpaint"
        ]

        result = handoff(
            latent=torch.zeros((1, 2, 1, 1, 1)),
            source=torch.zeros((3, 2, 4, 6)),
            generation_mask=torch.ones(
                (1, 2, 4, 6),
                dtype=torch.uint8,
            ),
            video_decoder=object(),
            video_encoder=object(),
            tiling_config=None,
            num_frames=2,
            height=4,
            width=6,
            target_height=8,
            target_width=12,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertIs(result, expected_latent)
        self.assertEqual(calls, ["decode", "blend", "encode"])
        self.assertEqual(
            tuple(captured["video"].shape),
            (1, 3, 2, 8, 12),
        )
        self.assertTrue(
            torch.allclose(
                captured["video"],
                torch.full_like(
                    captured["video"],
                    127.0 / 127.5 - 1.0,
                ),
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_generated_canvas_matches_source_render_without_touching_source(self):
        import torch
        import torch.nn.functional as F
        from models.ltx2.inpainting import _apply_ltx2_mask_blend

        height, width = 96, 128
        y = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1)
        x = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
        texture = (
            torch.sin(x * math.pi * 12.0)
            * torch.cos(y * math.pi * 8.0)
            * 0.08
        )
        target_unit = torch.cat(
            (
                (0.20 + x * 0.55 + texture).expand(1, 2, height, width),
                (0.18 + y * 0.60 - texture).expand(1, 2, height, width),
                (0.25 + (x + y) * 0.30 + texture).expand(
                    1,
                    2,
                    height,
                    width,
                ),
            ),
            dim=0,
        ).clamp(0.0, 1.0)
        frames_nchw = target_unit.permute(1, 0, 2, 3)
        softened = F.avg_pool2d(
            F.pad(frames_nchw, (1, 1, 1, 1), mode="replicate"),
            kernel_size=3,
            stride=1,
        ).permute(1, 0, 2, 3)
        generated_unit = (softened * 0.78).clamp(0.0, 1.0)
        source = target_unit.mul(2.0).sub(1.0)
        generated = generated_unit.mul(2.0).sub(1.0)
        mask = torch.ones((1, 2, height, width), dtype=torch.uint8)
        mask[:, :, 24:72, 8:120] = 0

        result = _apply_ltx2_mask_blend(
            generated,
            source,
            mask,
            2,
            height,
            width,
            mask_low_res_dilation=6,
            source_feather_pixels=8,
        )
        result_unit = result.add(1.0).mul(0.5)
        canvas = mask.expand_as(result_unit) > 0
        before_error = (
            generated_unit[canvas] - target_unit[canvas]
        ).square().mean()
        after_error = (
            result_unit[canvas] - target_unit[canvas]
        ).square().mean()

        self.assertLess(float(after_error), float(before_error) * 0.5)
        self.assertTrue(
            torch.equal(
                result[:, :, 36:60, 24:104],
                source[:, :, 36:60, 24:104],
            )
        )

    def test_generated_canvas_removes_mask_marker_chroma_without_tinting_black(self):
        import torch
        from models.ltx2.inpainting import _apply_ltx2_mask_blend

        height, width = 80, 112
        y = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1)
        x = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
        source_unit = torch.cat(
            (
                (0.30 + x * 0.25).expand(1, 2, height, width),
                (0.32 + y * 0.20).expand(1, 2, height, width),
                (0.28 + (x + y) * 0.075).expand(
                    1,
                    2,
                    height,
                    width,
                ),
            ),
            dim=0,
        )
        source_unit[:, :, :8, :8] = 0.0

        red_weight = 0.2126
        green_weight = 0.7152
        blue_weight = 0.0722
        source_luma = (
            source_unit[0] * red_weight
            + source_unit[1] * green_weight
            + source_unit[2] * blue_weight
        )
        light_weight = (
            (source_luma - 0.03).div(0.22).clamp(0.0, 1.0)
        )
        light_weight = light_weight.square() * (3.0 - 2.0 * light_weight)
        # Reproduce a luminance-preserving yellow/green marker spill:
        # too much red chroma and substantially too little blue chroma.
        tinted_red = (
            source_luma
            + (source_unit[0] - source_luma)
            + 0.015 * light_weight
        )
        tinted_blue = (
            source_luma
            + (source_unit[2] - source_luma)
            - 0.090 * light_weight
        )
        tinted_green = (
            source_luma
            - red_weight * tinted_red
            - blue_weight * tinted_blue
        ).div(green_weight)
        generated_unit = torch.stack(
            (tinted_red, tinted_green, tinted_blue),
            dim=0,
        ).clamp(0.0, 1.0)

        source = source_unit.mul(2.0).sub(1.0)
        generated = generated_unit.mul(2.0).sub(1.0)
        mask = torch.ones((1, 2, height, width), dtype=torch.uint8)
        mask[:, :, 16:64, 8:104] = 0

        result = _apply_ltx2_mask_blend(
            generated,
            source,
            mask,
            2,
            height,
            width,
            mask_low_res_dilation=6,
            source_feather_pixels=8,
        )
        result_unit = result.add(1.0).mul(0.5)

        def opponent_chroma(video):
            luma = (
                video[0] * red_weight
                + video[1] * green_weight
                + video[2] * blue_weight
            )
            return torch.stack(
                (video[0] - luma, video[2] - luma),
                dim=0,
            )

        target_chroma = opponent_chroma(source_unit)
        generated_chroma = opponent_chroma(generated_unit)
        result_chroma = opponent_chroma(result_unit)
        canvas = mask.expand(2, -1, -1, -1) > 0
        before_error = (
            generated_chroma[canvas] - target_chroma[canvas]
        ).square().mean()
        after_error = (
            result_chroma[canvas] - target_chroma[canvas]
        ).square().mean()

        self.assertLess(float(after_error), float(before_error) * 0.15)
        self.assertLess(float(result_unit[:, :, :6, :6].abs().max()), 1e-6)
        self.assertTrue(
            torch.equal(
                result[:, :, 28:52, 24:88],
                source[:, :, 28:52, 24:88],
            )
        )

    def test_generated_only_marker_spill_uses_source_boundary_chroma(self):
        import torch
        from models.ltx2.inpainting import (
            _apply_ltx2_mask_blend,
            _estimate_side_chroma_transfers,
        )

        height, width = 80, 112
        y = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1)
        x = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
        source_unit = torch.cat(
            (
                (0.24 + x * 0.28).expand(1, 3, height, width),
                (0.27 + y * 0.20).expand(1, 3, height, width),
                (0.22 + (x + y) * 0.10).expand(
                    1,
                    3,
                    height,
                    width,
                ),
            ),
            dim=0,
        )
        mask = torch.ones((1, 3, height, width), dtype=torch.uint8)
        mask[:, :, 16:64, 8:104] = 0
        generated_unit = source_unit.clone()

        red_weight = 0.2126
        green_weight = 0.7152
        blue_weight = 0.0722
        luma = (
            source_unit[0] * red_weight
            + source_unit[1] * green_weight
            + source_unit[2] * blue_weight
        )
        tinted_red = luma + (source_unit[0] - luma) + 0.020
        tinted_blue = luma + (source_unit[2] - luma) - 0.080
        tinted_green = (
            luma
            - red_weight * tinted_red
            - blue_weight * tinted_blue
        ).div(green_weight)
        tinted = torch.stack(
            (tinted_red, tinted_green, tinted_blue),
            dim=0,
        ).clamp(0.0, 1.0)
        canvas = mask.expand_as(generated_unit) > 0
        generated_unit[canvas] = tinted[canvas]

        source = source_unit.mul(2.0).sub(1.0)
        generated = generated_unit.mul(2.0).sub(1.0)
        transfers = _estimate_side_chroma_transfers(
            generated,
            source,
            mask,
            gain=1.0,
            offset=0.0,
        )
        self.assertTrue(all(transfer is None for transfer in transfers))
        result = _apply_ltx2_mask_blend(
            generated,
            source,
            mask,
            3,
            height,
            width,
            mask_low_res_dilation=6,
            source_feather_pixels=8,
        )
        result_unit = result.add(1.0).mul(0.5)

        def opponent_chroma(video):
            video_luma = (
                video[0] * red_weight
                + video[1] * green_weight
                + video[2] * blue_weight
            )
            return torch.stack(
                (video[0] - video_luma, video[2] - video_luma),
                dim=0,
            )

        chroma_canvas = mask.expand(2, -1, -1, -1) > 0
        target_chroma = opponent_chroma(source_unit)
        before_chroma = opponent_chroma(generated_unit)
        after_chroma = opponent_chroma(result_unit)
        before_error = (
            before_chroma[chroma_canvas] - target_chroma[chroma_canvas]
        ).square().mean()
        after_error = (
            after_chroma[chroma_canvas] - target_chroma[chroma_canvas]
        ).square().mean()

        self.assertLess(float(after_error), float(before_error) * 0.20)
        self.assertTrue(
            torch.equal(
                result[:, :, 28:52, 24:88],
                source[:, :, 28:52, 24:88],
            )
        )

    def test_low_correlation_window_cannot_reverse_boundary_chroma_fix(self):
        import torch
        from models.ltx2.inpainting import _estimate_canvas_match

        height, width = 80, 112
        y = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1)
        x = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
        source_unit = torch.cat(
            (
                (0.22 + x * 0.40).expand(1, 3, height, width),
                (0.20 + y * 0.45).expand(1, 3, height, width),
                (0.24 + (x + y) * 0.16).expand(
                    1,
                    3,
                    height,
                    width,
                ),
            ),
            dim=0,
        ).clamp(0.0, 1.0)
        mask = torch.ones((1, 3, height, width), dtype=torch.uint8)
        mask[:, :, 16:64, 8:104] = 0
        generated_unit = source_unit.clone()

        luma = (
            source_unit[0] * 0.2126
            + source_unit[1] * 0.7152
            + source_unit[2] * 0.0722
        )
        tinted_red = luma + (source_unit[0] - luma) + 0.015
        tinted_blue = luma + (source_unit[2] - luma) - 0.075
        tinted_green = (
            luma - 0.2126 * tinted_red - 0.0722 * tinted_blue
        ).div(0.7152)
        tinted = torch.stack(
            (tinted_red, tinted_green, tinted_blue),
            dim=0,
        ).clamp(0.0, 1.0)
        canvas = mask.expand_as(generated_unit) > 0
        generated_unit[canvas] = tinted[canvas]
        generated_unit[:, :, 16:64, 8:104] = (
            1.0 - source_unit[:, :, 16:64, 8:104]
        )

        match = _estimate_canvas_match(
            generated_unit.mul(2.0).sub(1.0),
            source_unit.mul(2.0).sub(1.0),
            mask,
            source_inset=8,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match[0], 1.0)
        self.assertEqual(match[1], 0.0)
        self.assertGreater(match[3], 0.03)
        self.assertGreaterEqual(match[7], 2)

    def test_marker_spill_growing_away_from_one_seam_is_corrected_per_side(self):
        import torch
        from models.ltx2.inpainting import (
            _apply_ltx2_mask_blend,
            _estimate_side_chroma_transfers,
        )

        height, width = 160, 112
        y = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1)
        x = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
        source_unit = torch.cat(
            (
                (0.34 + x * 0.12).expand(1, 4, height, width),
                (0.36 + y * 0.10).expand(1, 4, height, width),
                (0.27 + (x + y) * 0.055).expand(
                    1,
                    4,
                    height,
                    width,
                ),
            ),
            dim=0,
        )
        mask = torch.ones((1, 4, height, width), dtype=torch.uint8)
        mask[:, :, 48:112, 8:104] = 0

        red_weight = 0.2126
        green_weight = 0.7152
        blue_weight = 0.0722
        luma = (
            source_unit[0] * red_weight
            + source_unit[1] * green_weight
            + source_unit[2] * blue_weight
        )
        bottom_distance = (
            (torch.arange(height, dtype=torch.float32) - 111.0)
            .div(height - 112)
            .clamp(0.0, 1.0)
            .sqrt()
            .view(1, height, 1)
        )
        blue_deficit = 0.020 + bottom_distance * 0.160
        tinted_blue = luma + (source_unit[2] - luma) - blue_deficit
        tinted_red = source_unit[0]
        tinted_green = (
            luma
            - red_weight * tinted_red
            - blue_weight * tinted_blue
        ).div(green_weight)
        tinted = torch.stack(
            (tinted_red, tinted_green, tinted_blue),
            dim=0,
        ).clamp(0.0, 1.0)
        generated_unit = source_unit.clone()
        canvas = mask.expand_as(generated_unit) > 0
        generated_unit[canvas] = tinted[canvas]

        source = source_unit.mul(2.0).sub(1.0)
        generated = generated_unit.mul(2.0).sub(1.0)
        transfers = _estimate_side_chroma_transfers(
            generated,
            source,
            mask,
            gain=1.0,
            offset=0.0,
        )
        self.assertIsNone(transfers[0])
        self.assertIsNotNone(transfers[1])
        result = _apply_ltx2_mask_blend(
            generated,
            source,
            mask,
            4,
            height,
            width,
            mask_low_res_dilation=6,
            source_feather_pixels=8,
        )
        result_unit = result.add(1.0).mul(0.5)

        def blue_chroma(video):
            video_luma = (
                video[0] * red_weight
                + video[1] * green_weight
                + video[2] * blue_weight
            )
            return video[2] - video_luma

        target = blue_chroma(source_unit)[:, -24:]
        before = blue_chroma(generated_unit)[:, -24:]
        after = blue_chroma(result_unit)[:, -24:]
        before_error = (before - target).square().mean()
        after_error = (after - target).square().mean()
        target_iqr = torch.quantile(target, 0.75) - torch.quantile(
            target,
            0.25,
        )
        before_iqr = torch.quantile(before, 0.75) - torch.quantile(
            before,
            0.25,
        )
        after_iqr = torch.quantile(after, 0.75) - torch.quantile(
            after,
            0.25,
        )

        self.assertLess(float(after_error), float(before_error) * 0.20)
        self.assertLess(
            abs(float(after_iqr - target_iqr)),
            abs(float(before_iqr - target_iqr)) * 0.35,
        )
        self.assertTrue(
            torch.equal(
                result[:, :, 64:96, 24:88],
                source[:, :, 64:96, 24:88],
            )
        )

    def test_single_stage_blend_leaves_generated_canvas_unfiltered(self):
        import torch
        from models.ltx2.inpainting import _apply_ltx2_mask_blend

        height, width = 64, 96
        x = torch.linspace(-0.8, 0.8, width).view(1, 1, 1, width)
        generated = x.expand(3, 2, height, width).clone()
        source = generated.mul(0.5).add(0.25)
        mask = torch.ones((1, 2, height, width), dtype=torch.uint8)
        mask[:, :, 16:48, 8:88] = 0

        result = _apply_ltx2_mask_blend(
            generated,
            source,
            mask,
            2,
            height,
            width,
            mask_low_res_dilation=6,
            source_feather_pixels=8,
            match_generated_canvas=False,
        )

        self.assertTrue(
            torch.allclose(
                result[:, :, :12, :],
                generated[:, :, :12, :],
                atol=1e-6,
                rtol=0.0,
            )
        )
        self.assertTrue(
            torch.allclose(
                result[:, :, 28:36, 24:72],
                source[:, :, 28:36, 24:72],
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_official_blend_removes_only_detected_marker_gradient(self):
        import torch
        from models.ltx2.inpainting import _apply_ltx2_mask_blend

        height, width = 160, 96
        frames = 4
        x = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
        y = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1)
        source_unit = torch.cat(
            (
                (0.34 + x * 0.10).expand(1, frames, height, width),
                (0.30 + y * 0.08).expand(1, frames, height, width),
                (0.20 + x * 0.05).expand(1, frames, height, width),
            ),
            dim=0,
        )
        mask = torch.ones((1, frames, height, width), dtype=torch.uint8)
        mask[:, :, 48:112] = 0

        luma = (
            source_unit[0] * 0.2126
            + source_unit[1] * 0.7152
            + source_unit[2] * 0.0722
        )
        bottom_ramp = (
            (torch.arange(height, dtype=torch.float32) - 111.0)
            .div(height - 112)
            .clamp(0.0, 1.0)
            .sqrt()
            .view(1, height, 1)
        )
        tinted_blue = luma + (source_unit[2] - luma) - bottom_ramp * 0.16
        tinted_red = source_unit[0]
        tinted_green = (
            luma - 0.2126 * tinted_red - 0.0722 * tinted_blue
        ).div(0.7152)
        tinted = torch.stack(
            (tinted_red, tinted_green, tinted_blue),
            dim=0,
        ).clamp(0.0, 1.0)
        generated_unit = source_unit.clone()
        canvas = mask.expand_as(generated_unit) > 0
        generated_unit[canvas] = tinted[canvas]

        source = source_unit.mul(2.0).sub(1.0)
        generated = generated_unit.mul(2.0).sub(1.0)
        baseline = _apply_ltx2_mask_blend(
            generated,
            source,
            mask,
            frames,
            height,
            width,
            mask_low_res_dilation=2,
            source_feather_pixels=8,
            match_generated_canvas=False,
            full_frame_laplacian=True,
            correct_marker_residue=False,
        )
        result = _apply_ltx2_mask_blend(
            generated,
            source,
            mask,
            frames,
            height,
            width,
            mask_low_res_dilation=2,
            source_feather_pixels=8,
            match_generated_canvas=False,
            full_frame_laplacian=True,
            correct_marker_residue=True,
        )
        result_unit = result.add(1.0).mul(0.5)

        def blue_chroma(video):
            video_luma = (
                video[0] * 0.2126
                + video[1] * 0.7152
                + video[2] * 0.0722
            )
            return video[2] - video_luma

        target = blue_chroma(source_unit)[:, -24:]
        before = blue_chroma(generated_unit)[:, -24:]
        after = blue_chroma(result_unit)[:, -24:]
        self.assertLess(
            float((after - target).square().mean()),
            float((before - target).square().mean()) * 0.40,
        )
        self.assertTrue(
            torch.equal(
                result[:, :, 64:96, 16:80],
                baseline[:, :, 64:96, 16:80],
            )
        )

    def test_feather_factories_inherit_the_cpu_mask_device(self):
        import torch
        from models.ltx2 import inpainting

        mask = torch.ones(
            (1, 1, 96, 128),
            dtype=torch.uint8,
            device="cpu",
        )
        mask[:, :, 24:72, 16:112] = 0
        original_full = torch.full
        original_arange = torch.arange

        def require_device(factory):
            def guarded(*args, **kwargs):
                self.assertEqual(kwargs.get("device"), mask.device)
                return factory(*args, **kwargs)

            return guarded

        # Maestro's inference wrapper installs CUDA as PyTorch's default
        # device. Any unqualified factory here would therefore create a CUDA
        # tensor beside the deliberately CPU-resident postprocess mask.
        with (
            mock.patch.object(
                inpainting.torch,
                "full",
                side_effect=require_device(original_full),
            ),
            mock.patch.object(
                inpainting.torch,
                "arange",
                side_effect=require_device(original_arange),
            ),
        ):
            alpha = inpainting._build_source_boundary_feather_alpha(
                mask,
                8,
            )
            gaussian_alpha = inpainting._build_gaussian_margin_alpha(
                mask,
                8,
            )

        self.assertEqual(alpha.device, mask.device)
        self.assertEqual(gaussian_alpha.device, mask.device)

    def test_canvas_match_sampling_inherits_the_cpu_mask_device(self):
        import torch
        from models.ltx2 import inpainting

        height, width = 64, 96
        source = torch.linspace(
            -1.0,
            1.0,
            3 * 2 * height * width,
            device="cpu",
        ).reshape(3, 2, height, width)
        generated = source.mul(0.8)
        mask = torch.ones(
            (1, 2, height, width),
            dtype=torch.uint8,
            device="cpu",
        )
        mask[:, :, 12:52, 8:88] = 0
        original_linspace = torch.linspace

        def guarded_linspace(*args, **kwargs):
            self.assertEqual(kwargs.get("device"), mask.device)
            return original_linspace(*args, **kwargs)

        with mock.patch.object(
            inpainting.torch,
            "linspace",
            side_effect=guarded_linspace,
        ):
            match = inpainting._estimate_canvas_match(
                generated,
                source,
                mask,
                source_inset=8,
            )

        self.assertIsNotNone(match)

    def test_ui_replaces_misleading_controls_with_one_recommended_toggle(self):
        controls = _read(_OUTPAINT_CONTROLS_PATH)
        client = _read(_CLIENT_PATH)
        store = _read(_STORE_PATH)
        self.assertIn("Preserve original scene", controls)
        self.assertIn("outpaintMaskPreserving: true", store)
        self.assertIn("mask_preserving_outpaint?: boolean", client)
        self.assertIn("outpaint_aspect?:", client)
        self.assertIn("outpaint_aspect: state.outpaintAspect", store)
        self.assertIn("_inferOutpaintAspect", store)
        self.assertIn("source_preservation: 1.0", store)
        self.assertIn("outpaint_lora_strength: 1.0", store)
        self.assertIn("lock_source_pixels: false", store)
        self.assertNotIn("Outpaint LoRA Strength", controls)
        self.assertNotIn("Lock source pixels", controls)
        self.assertNotIn(">Source Preservation<", controls)

    def test_canvas_readout_tracks_quality_preset_and_model_grid(self):
        canvas = _read(_OUTPAINT_CANVAS_PATH)
        self.assertIn("OUTPUT_PIXEL_BUDGETS", canvas)
        self.assertIn("resolutionPreset", canvas)
        self.assertIn("resolvedCanvasPx", canvas)
        self.assertIn("Math.round(value / alignment) * alignment", canvas)
        self.assertIn(
            "Canvas: {resolvedCanvasPx.w}×{resolvedCanvasPx.h}px",
            canvas,
        )

    def test_outpaint_generate_button_explains_zero_generation_area(self):
        generate_button = _read(
            os.path.join(
                _ROOT,
                "ui",
                "src",
                "components",
                "Sidebar",
                "GenerateButton.tsx",
            )
        )
        self.assertIn("needsOutpaintArea", generate_button)
        self.assertIn("Choose canvas", generate_button)
        self.assertIn("area for Outpaint to generate", generate_button)

    def test_backend_uses_official_lora_and_internal_blend(self):
        launch = _read(_LAUNCH_PATH)
        wgp = _read(_WGP_PATH)
        ltx2 = _read(_LTX2_PATH)
        distilled = _read(_LTX2_DISTILLED_PATH)
        inpainting = _read(_LTX2_INPAINT_PATH)
        self.assertIn('"outpaint_mask_preserve": mask_preserving_outpaint', launch)
        self.assertIn(
            '"outpaint_official_stack": official_outpaint',
            launch,
        )
        self.assertNotIn(
            "reference_fps=24.0 if official_outpaint else None",
            launch,
        )
        self.assertIn(
            '"force_fps": "control" if is_video else "auto"',
            launch,
        )
        self.assertIn(
            "_resolve_ltx2_outpaint_reference_model(model_type)",
            launch,
        )
        self.assertIn(
            '"[Outpaint] Reference sampling: 8-step masked first pass + "',
            launch,
        )
        self.assertIn("guidance_scale = 1.0", launch)
        self.assertIn(
            "guide_inpaint_color = (128, 128, 128)",
            wgp,
        )
        shared_preprocess_block = wgp.split(
            "if (\n        outpaint_mask_preserve",
            1,
        )[1].split(
            "# Aspect-ratio outpainting",
            1,
        )[0]
        self.assertNotIn("(102, 255, 0)", shared_preprocess_block)
        self.assertIn(
            "if outpaint_official_stack:",
            wgp,
        )
        self.assertIn(
            "outpaint_official_stack=outpaint_official_stack",
            wgp,
        )
        self.assertIn(
            "LTX-2.3-22b-IC-LoRA-In-Outpainting",
            ltx2,
        )
        self.assertIn('"1;1"', ltx2)
        self.assertIn(
            "active_official_outpaint = None",
            ltx2,
        )
        self.assertIn("_apply_ltx2_mask_blend(", ltx2)
        self.assertIn("_build_gaussian_margin_alpha", inpainting)
        self.assertIn("chunk_frames", inpainting)
        self.assertIn(
            '"single_stage_pipeline": False',
            launch,
        )
        self.assertIn(
            '"outpaint_full_resolution_refine": official_outpaint',
            launch,
        )
        self.assertIn(
            "outpaint_full_resolution_refine="
            "outpaint_full_resolution_refine",
            wgp,
        )
        self.assertIn("full_resolution_refine", distilled)
        self.assertIn(
            "OUTPAINT_ATTENTION_STAGE_2_SIGMA_VALUES",
            distilled,
        )
        self.assertIn("LTX2_MASKED_CONTROL_VIDEO_PAD_RGB", inpainting)
        self.assertIn("_paint_ltx2_masked_control_video(", ltx2)
        self.assertIn("_paint_ltx2_inpaint_control_video(", ltx2)
        self.assertIn(
            "_outpaint_source_attention_to_latents(",
            _read(_LTX2_HELPERS_PATH),
        )
        self.assertIn(
            'match_generated_canvas=not bool(',
            ltx2,
        )
        self.assertIn(
            "full_frame_laplacian=production_outpaint_blend",
            ltx2,
        )
        self.assertIn(
            "2\n                    if production_outpaint_blend",
            ltx2,
        )
        official_sampler_branch = distilled.split(
            "if full_resolution_refine:",
            1,
        )[1].split("elif single_stage:", 1)[0]
        self.assertIn(
            "stepper = EulerDiffusionStep()",
            official_sampler_branch,
        )
        self.assertNotIn(
            "stepper = EulerAncestralDiffusionStep(",
            official_sampler_branch,
        )
        refinement_sampler_branch = distilled.split(
            "# Standard stage 2 uses ancestral sampling",
            1,
        )[1].split("self_refiner_handler", 1)[0]
        self.assertIn(
            "if full_resolution_refine:",
            refinement_sampler_branch,
        )
        self.assertIn(
            "stepper_stage2 = EulerDiffusionStep()",
            refinement_sampler_branch,
        )
        self.assertIn(
            "LTX2_OUTPAINTING_MARKER_RESIDUE_CLEANUP",
            ltx2,
        )
        self.assertIn(
            "correct_marker_residue=bool(",
            ltx2,
        )
        self.assertIn(
            "[LTX2] Official Lightricks Outpaint",
            ltx2,
        )
        official_branch = ltx2.split(
            "elif full_resolution_outpaint_refine:",
            1,
        )[1].split("else:", 1)[0]
        self.assertIn(
            "_paint_ltx2_inpaint_control_video",
            official_branch,
        )
        self.assertNotIn(
            "_paint_ltx2_masked_control_video",
            official_branch,
        )
        self.assertIn(
            "decoded-pixel Lanczos handoff",
            ltx2,
        )
        self.assertIn(
            "Lightricks recommends an empty/minimal prompt",
            launch,
        )
        self.assertNotIn(
            'f"{raw_outpaint_prompt}; seamlessly extend',
            launch.split("if official_outpaint:", 1)[1].split("else:", 1)[0],
        )
        self.assertIn(
            "reference_attention=",
            ltx2,
        )
        self.assertIn(
            "_select_reference_attention_generation_mask(",
            distilled,
        )
        self.assertIn(
            "generation_mask=reference_attention_generation_mask",
            distilled,
        )
        self.assertIn(
            "2-step decoded-pixel refinement",
            launch,
        )
        self.assertIn('mode="area"', distilled)
        self.assertIn(
            "and not full_resolution_outpaint_refine",
            ltx2,
        )


class TestLtx2OutpaintPipelineDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class FakeDistilledPipeline:
            pass

        class FakeTwoStagePipeline:
            pass

        cls.fake_distilled_pipeline = FakeDistilledPipeline
        cls.fake_two_stage_pipeline = FakeTwoStagePipeline
        cls.uses_distilled = staticmethod(
            _load_functions(
                _LTX2_PATH,
                ("_uses_distilled_pipeline_dispatch",),
                {"DistilledPipeline": FakeDistilledPipeline},
            )["_uses_distilled_pipeline_dispatch"]
        )
        cls.uses_native_two_stage = staticmethod(
            _load_functions(
                _LTX2_PATH,
                ("_uses_native_two_stage_dispatch",),
                {"TI2VidTwoStagesPipeline": FakeTwoStagePipeline},
            )["_uses_native_two_stage_dispatch"]
        )
        cls.select_generation_mask = staticmethod(
            _load_functions(
                _LTX2_PATH,
                ("_select_video_conditioning_generation_mask",),
            )["_select_video_conditioning_generation_mask"]
        )
        cls.select_reference_attention_mask = staticmethod(
            _load_functions(
                _LTX2_DISTILLED_PATH,
                ("_select_reference_attention_generation_mask",),
                {"torch": types.SimpleNamespace(Tensor=object)},
            )["_select_reference_attention_generation_mask"]
        )

    def test_native_distilled_pipeline_is_recognized(self):
        self.assertTrue(
            self.uses_distilled(
                self.fake_distilled_pipeline(),
                None,
            )
        )

    def test_explicit_dev_outpaint_adapter_is_authoritative(self):
        wrapped_pipeline = object()
        official_adapter = object()
        self.assertTrue(
            self.uses_distilled(
                wrapped_pipeline,
                official_adapter,
            )
        )

    def test_unrelated_pipeline_is_not_distilled(self):
        self.assertFalse(self.uses_distilled(object(), None))

    def test_native_dev_pipeline_runs_without_official_adapter(self):
        self.assertTrue(
            self.uses_native_two_stage(
                self.fake_two_stage_pipeline(),
                None,
            )
        )

    def test_official_adapter_bypasses_native_dev_pipeline(self):
        self.assertFalse(
            self.uses_native_two_stage(
                self.fake_two_stage_pipeline(),
                object(),
            )
        )

    def test_official_dispatch_selects_the_primary_input_mask(self):
        primary_mask = object()
        fallback_mask = object()
        self.assertIs(
            self.select_generation_mask(
                True,
                object(),
                primary_mask,
                fallback_mask,
            ),
            primary_mask,
        )

    def test_official_dispatch_selects_the_fallback_input_mask(self):
        primary_mask = object()
        fallback_mask = object()
        self.assertIs(
            self.select_generation_mask(
                True,
                None,
                primary_mask,
                fallback_mask,
            ),
            fallback_mask,
        )

    def test_non_outpaint_dispatch_has_no_generation_mask(self):
        self.assertIsNone(
            self.select_generation_mask(
                False,
                object(),
                object(),
                object(),
            )
        )

    def test_official_pixel_refine_uses_full_reference_attention(self):
        generation_mask = object()
        self.assertIsNone(
            self.select_reference_attention_mask(
                True,
                generation_mask,
            )
        )

    def test_diffusers_reference_path_retains_source_attention(self):
        generation_mask = object()
        self.assertIs(
            self.select_reference_attention_mask(
                False,
                generation_mask,
            ),
            generation_mask,
        )

    def test_managed_outpaint_lora_remains_on_for_both_passes(self):
        app_root = os.path.join(_ROOT, "app")
        if app_root not in sys.path:
            sys.path.insert(0, app_root)
        from shared.utils.loras_mutipliers import (
            expand_slist,
            parse_loras_multipliers,
        )

        initial, phases, error = parse_loras_multipliers(
            "1;1",
            1,
            8,
            nb_phases=2,
        )
        self.assertEqual(error, "")
        self.assertEqual(initial, [1.0])
        self.assertEqual(
            expand_slist(phases, 0, 8, 8, 8),
            1.0,
        )
        self.assertEqual(
            expand_slist(phases, 0, 3, 0, 3),
            1.0,
        )

    def test_official_dispatch_forwards_local_refinement_flag(self):
        source = _read(_LTX2_PATH)
        self.assertIn(
            "_uses_distilled_pipeline_dispatch(",
            source,
        )
        self.assertIn(
            "full_resolution_refine=(\n"
            "                        full_resolution_outpaint_refine",
            source,
        )
        dispatch_block = source.split(
            "if _uses_distilled_pipeline_dispatch(",
            1,
        )[1].split("else:", 1)[0]
        self.assertLess(
            dispatch_block.index(
                "video_conditioning_generation_mask ="
            ),
            dispatch_block.index("generation_mask="),
        )
        self.assertGreaterEqual(
            source.count("_uses_native_two_stage_dispatch("),
            3,
        )


class TestRetakePipelineCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolve = staticmethod(
            _load_functions(
                _LTX2_PATH,
                ("_resolve_retake_pipeline_models",),
            )["_resolve_retake_pipeline_models"]
        )

    def test_distilled_pipeline_models_are_supported(self):
        models = object()
        pipeline = types.SimpleNamespace(models=models)
        self.assertIs(self.resolve(pipeline), models)

    def test_two_stage_pipeline_uses_stage_one_models(self):
        stage_one = object()
        stage_two = object()
        pipeline = types.SimpleNamespace(
            stage_1_models=stage_one,
            stage_2_models=stage_two,
        )
        self.assertIs(self.resolve(pipeline), stage_one)

    def test_unsupported_pipeline_has_a_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "does not expose models"):
            self.resolve(types.SimpleNamespace())

    def test_native_retake_uses_the_compatibility_resolver(self):
        source = _read(_LTX2_PATH)
        self.assertIn(
            "retake_models = _resolve_retake_pipeline_models(self.pipeline)",
            source,
        )
        self.assertIn("models=retake_models", source)
        self.assertNotIn(
            "RetakePipeline(\n                models=self.pipeline.models",
            source,
        )


class TestSingleClipDirectorOutput(unittest.TestCase):
    def test_single_item_group_is_not_concatenated(self):
        source = _read(_WGP_PATH)
        self.assertIn(
            'int(multi_clip_info.get("total", 0) or 0) > 1',
            source,
        )


class TestModelVisibilityPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normalize = staticmethod(
            _load_functions(
                _LAUNCH_PATH,
                ("_normalize_model_visibility_ids",),
            )["_normalize_model_visibility_ids"]
        )

    def test_model_ids_are_trimmed_and_deduplicated(self):
        self.assertEqual(
            self.normalize([
                "flux2_klein_4b",
                " flux2_klein_4b ",
                "",
                "ltx2_22B_10eros",
            ]),
            ["flux2_klein_4b", "ltx2_22B_10eros"],
        )

    def test_invalid_visibility_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            self.normalize("flux2_klein_4b")
        with self.assertRaises(ValueError):
            self.normalize([123])

    def test_server_and_frontend_use_durable_visibility(self):
        launch = _read(_LAUNCH_PATH)
        client = _read(_CLIENT_PATH)
        store = _read(_STORE_PATH)
        self.assertIn('@api.get("/api/v1/model-visibility")', launch)
        self.assertIn('@api.put("/api/v1/model-visibility")', launch)
        self.assertIn("os.replace(temp_path, config_path)", launch)
        self.assertIn("fetchModelVisibility", client)
        self.assertIn("updateModelVisibility", client)
        self.assertIn("api.fetchModelVisibility()", store)
        self.assertIn("api.updateModelVisibility(payload)", store)
        self.assertIn("initialized_mature_models", store)
        self.assertIn("_enableUninitializedMatureModels", store)
        self.assertNotIn("for (const mt of nsfwModels)", store)


class TestFramesControlVideoAudio(unittest.TestCase):
    def test_control_video_presence_is_not_derived_from_audio_mode(self):
        source = _read(_INPUTS_PATH)
        self.assertIn(
            "const hasControlVid = supportsControlVid && "
            "!!params.video_guide",
            source,
        )
        self.assertIn("Generate soundtrack from text prompt", source)
        self.assertIn("Generate new audio from control video", source)
        self.assertIn("The control video remains attached", source)
        self.assertIn("rawControlProcess", source)
        self.assertNotIn("supportsSoundtrack && !hasControlVid", source)
        self.assertNotIn("supportsControlVid && !hasSoundtrack", source)


if __name__ == "__main__":
    unittest.main()
