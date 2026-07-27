"""Model-free regressions for SCAIL-2 Recast and Repaint workflows."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import tempfile
import unittest

import numpy as np


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LAUNCH_PATH = os.path.join(_ROOT, "app", "launch.py")
_WGP_PATH = os.path.join(_ROOT, "app", "wgp.py")
_WAN_HANDLER_PATH = os.path.join(
    _ROOT, "app", "models", "wan", "wan_handler.py",
)
_SCAIL2_PATH = os.path.join(_ROOT, "app", "models", "wan", "scail2", "__init__.py")
_UTILS_PATH = os.path.join(_ROOT, "app", "shared", "utils", "utils.py")
_SCAIL2_FAST_PATH = os.path.join(_ROOT, "app", "defaults", "scail2_14B_fast.json")
_SCAIL2_RECAST_FAST_PATH = os.path.join(
    _ROOT, "app", "defaults", "scail2_14B_recast_fast.json",
)
_SCAIL2_LORA_PATH = os.path.join(
    _ROOT, "app", "services", "scail2_lora.py",
)
_SAM3_PREPROCESSOR_PATH = os.path.join(
    _ROOT, "app", "preprocessing", "sam3", "preprocessor.py",
)
_SAM3_MULTIPLEX_DETECTOR_PATH = os.path.join(
    _ROOT,
    "app",
    "preprocessing",
    "sam3",
    "model",
    "sam3_multiplex_detector.py",
)
_MAGIC_MASK_PATH = os.path.join(
    _ROOT, "app", "shared", "magic_mask.py",
)
_LORA_MULTIPLIERS_PATH = os.path.join(
    _ROOT, "app", "shared", "utils", "loras_mutipliers.py",
)
_STORE_PATH = os.path.join(_ROOT, "ui", "src", "stores", "useStore.ts")
_RECAST_CONTROLS_PATH = os.path.join(
    _ROOT, "ui", "src", "components", "Sidebar", "RecastControls.tsx",
)
_REPAINT_CONTROLS_PATH = os.path.join(
    _ROOT, "ui", "src", "components", "Sidebar", "RestyleControls.tsx",
)
_SCAIL_RESOLUTION_SELECTOR_PATH = os.path.join(
    _ROOT, "ui", "src", "components", "Sidebar",
    "ScailResolutionSelector.tsx",
)
_INFO_TOOLTIP_PATH = os.path.join(
    _ROOT, "ui", "src", "components", "Sidebar", "InfoTooltip.tsx",
)
_PROMPT_INPUT_PATH = os.path.join(
    _ROOT, "ui", "src", "components", "Sidebar", "PromptInput.tsx",
)
_ADVANCED_SETTINGS_PATH = os.path.join(
    _ROOT, "ui", "src", "components", "Sidebar", "AdvancedSettings.tsx",
)
_EDIT_SUBMODE_PATH = os.path.join(
    _ROOT, "ui", "src", "components", "Sidebar", "EditSubModeToggle.tsx",
)
_API_CLIENT_PATH = os.path.join(_ROOT, "ui", "src", "api", "client.ts")
_LORA_SELECTOR_PATH = os.path.join(
    _ROOT, "ui", "src", "components", "SettingsDrawer", "LoraSelector.tsx",
)


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_functions(path: str, names: tuple[str, ...], namespace=None) -> dict:
    source = _read(path)
    tree = ast.parse(source, filename=os.path.relpath(path, _ROOT))
    selected = [
        node for node in tree.body
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


class TestScail2ProcessSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = _load_functions(
            _SCAIL2_PATH,
            (
                "test_scail2_replace",
                "_extract_max_people",
            ),
            {"re": re},
        )

    def test_animate_and_replace_codes_preserve_people_count(self):
        people = self.helpers["_extract_max_people"]
        replace = self.helpers["test_scail2_replace"]

        for count in range(1, 6):
            animate_code = f"V{count}AI"
            replace_code = f"V0{count}AI"
            self.assertEqual(people(animate_code), count)
            self.assertEqual(people(replace_code), count)
            self.assertFalse(replace(animate_code))
            self.assertTrue(replace(replace_code))


class TestScail2WindowSeedStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest("torch is required") from exc
        cls.torch = torch
        cls.get_generator = staticmethod(
            _load_functions(
                _SCAIL2_PATH,
                ("get_scail2_seed_generator",),
                {"torch": torch},
            )["get_scail2_seed_generator"]
        )

    def test_generator_advances_across_windows_like_official_segment_loop(self):
        class Pipeline:
            device = self.torch.device("cpu")

        pipeline = Pipeline()
        first_generator = self.get_generator(pipeline, 12345, 1)
        first_noise = self.torch.randn(16, generator=first_generator)
        second_generator = self.get_generator(pipeline, 12345, 2)
        second_noise = self.torch.randn(16, generator=second_generator)

        official_stream = self.torch.Generator(device="cpu")
        official_stream.manual_seed(12345)
        expected_first = self.torch.randn(16, generator=official_stream)
        expected_second = self.torch.randn(16, generator=official_stream)

        self.assertIs(first_generator, second_generator)
        self.assertTrue(self.torch.equal(first_noise, expected_first))
        self.assertTrue(self.torch.equal(second_noise, expected_second))
        self.assertFalse(self.torch.equal(first_noise, second_noise))

    def test_any2video_uses_continuous_stream_for_scail2_windows(self):
        any2video = _read(os.path.join(
            _ROOT, "app", "models", "wan", "any2video.py",
        ))
        self.assertIn(
            "get_scail2_seed_generator(self, seed, window_no)",
            any2video,
        )

    def test_first_window_resets_stream_for_a_new_generation(self):
        class Pipeline:
            device = self.torch.device("cpu")

        pipeline = Pipeline()
        initial = self.get_generator(pipeline, 777, 1)
        expected = self.torch.randn(8, generator=initial)
        continued = self.get_generator(pipeline, 777, 2)
        self.torch.randn(8, generator=continued)

        restarted = self.get_generator(pipeline, 777, 1)
        actual = self.torch.randn(8, generator=restarted)

        self.assertIsNot(restarted, initial)
        self.assertTrue(self.torch.equal(actual, expected))


class TestScail2RecastFastModel(unittest.TestCase):
    def test_recast_fast_uses_official_i2v_lightx_operating_point(self):
        with open(_SCAIL2_RECAST_FAST_PATH, "r", encoding="utf-8") as handle:
            recast = json.load(handle)
        with open(_SCAIL2_FAST_PATH, "r", encoding="utf-8") as handle:
            animate = json.load(handle)

        self.assertEqual(recast["model"]["architecture"], "scail2_14B")
        self.assertEqual(recast["model"]["URLs"], "scail2_14B")
        self.assertIn("lightx2v_I2V_14B_480p", recast["model"]["loras"][0])
        self.assertIn("rank128", recast["model"]["loras"][0])
        self.assertEqual(recast["model"]["loras_multipliers"], [1.0])
        self.assertEqual(recast["num_inference_steps"], 8)
        self.assertEqual(recast["flow_shift"], 1)
        self.assertEqual(recast["guidance_scale"], 1)
        self.assertEqual(recast["sample_solver"], "unipc")
        self.assertEqual(
            recast["custom_settings"]["scail2_recast_conditioning"],
            "native_replace",
        )
        # The proven Video/Animate Fast profile remains its separate rank-256
        # 6-step recipe; Recast no longer mutates it into a hybrid.
        self.assertIn("lightx2v_I2V_14B", animate["model"]["loras"][0])
        self.assertIn("rank256", animate["model"]["loras"][0])
        self.assertEqual(animate["num_inference_steps"], 6)


class TestScail2RelightingConverter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("scail2_lora_test", _SCAIL2_LORA_PATH)
        if spec is None or spec.loader is None:
            raise AssertionError("Could not load SCAIL-2 LoRA converter")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_official_sat_attention_and_mlp_keys_map_to_wan(self):
        convert = self.module.convert_key
        self.assertEqual(
            convert(
                "model.diffusion_model.transformer.layers.3.attention."
                "query_key_value.lora_layer.0.down.weight"
            ),
            "diffusion_model.blocks.3.self_attn.q.lora_down.weight",
        )
        self.assertEqual(
            convert(
                "model.diffusion_model.transformer.layers.7.cross_attention."
                "key_value.lora_layer.1.up.weight"
            ),
            "diffusion_model.blocks.7.cross_attn.v.lora_up.weight",
        )
        self.assertEqual(
            convert(
                "model.diffusion_model.transformer.layers.12.mlp."
                "dense_4h_to_h.lora_layer.diff_b"
            ),
            "diffusion_model.blocks.12.ffn.2.diff_b",
        )


class TestScail2RecastRelightingSettings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normalize = staticmethod(
            _load_functions(
                _LAUNCH_PATH,
                ("_normalize_recast_lora_settings",),
                {
                    "_RECAST_RELIGHTING_LORA_FILENAME":
                        "scail2_relighting_lora.safetensors",
                },
            )["_normalize_recast_lora_settings"]
        )
        spec = importlib.util.spec_from_file_location(
            "recast_lora_multipliers_test", _LORA_MULTIPLIERS_PATH,
        )
        if spec is None or spec.loader is None:
            raise AssertionError("Could not load LoRA multiplier parser")
        parser_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser_module)
        cls.parse = staticmethod(parser_module.parse_loras_multipliers)

    def test_main_toggle_ignores_orphan_three_phase_multiplier(self):
        loras, multipliers, active = self.normalize(
            [], "0.65;0.45;0.25", True,
        )

        self.assertEqual(loras, ["scail2_relighting_lora.safetensors"])
        self.assertEqual(multipliers, "1.0")
        self.assertTrue(active)
        self.assertEqual(
            self.parse(multipliers, len(loras), 8, nb_phases=1)[2],
            "",
        )

    def test_advanced_relighting_selection_keeps_first_phase_weight(self):
        loras, multipliers, active = self.normalize(
            ["scail2_relighting_lora.safetensors"],
            "0.70;0.50;0.30",
            False,
        )

        self.assertEqual(loras, ["scail2_relighting_lora.safetensors"])
        self.assertEqual(multipliers, "0.70")
        self.assertTrue(active)
        self.assertEqual(
            self.parse(multipliers, len(loras), 8, nb_phases=1)[2],
            "",
        )

    def test_recast_pairs_each_lora_before_collapsing_to_one_phase(self):
        loras, multipliers, active = self.normalize(
            [
                "style.safetensors",
                r"loras\wan_i2v\scail2_relighting_lora.safetensors",
                "style.safetensors",
            ],
            "0.80;0.60;0.40 0.55;0.45;0.35 0.20;0.10;0.05 orphan",
            True,
        )

        self.assertEqual(
            loras,
            [
                "style.safetensors",
                r"loras\wan_i2v\scail2_relighting_lora.safetensors",
            ],
        )
        self.assertEqual(multipliers, "0.80 0.55")
        self.assertNotIn(";", multipliers)
        self.assertTrue(active)


class TestScail2ControlVideoFraming(unittest.TestCase):
    def test_fixed_aspect_does_not_letterbox_scail_reference_before_crop(self):
        try:
            import torch
            import torch.nn.functional as functional
            from PIL import Image
        except ImportError as exc:
            raise unittest.SkipTest("torch or Pillow is unavailable") from exc

        resolve_fit = _load_functions(
            _WGP_PATH,
            ("_resolve_image_ref_fit",),
        )["_resolve_image_ref_fit"]
        center_crop = _load_functions(
            _SCAIL2_PATH,
            (
                "_resize_ref_image",
                "_center_crop_ref_image_to_canvas",
            ),
            {"F": functional},
        )["_center_crop_ref_image_to_canvas"]
        resize_refs = _load_functions(
            _UTILS_PATH,
            ("resize_and_remove_background",),
            {"np": np, "Image": Image},
        )["resize_and_remove_background"]

        scail_model = {
            "fit_into_canvas_image_refs": 0,
            "custom_image_ref_postprocessor_handles_canvas": True,
        }
        ref_fit = resolve_fit(scail_model, auto_aspect=False)
        self.assertEqual(ref_fit, 0)

        reference = Image.new("RGB", (1024, 1024), (17, 33, 65))
        resized_refs, _ = resize_refs(
            [reference],
            832,
            480,
            False,
            False,
            fit_into_canvas=ref_fit,
            return_tensor=False,
        )
        shared_ref = resized_refs[0]
        self.assertEqual(shared_ref.width, shared_ref.height)
        self.assertNotEqual(shared_ref.size, (832, 480))

        ref_array = np.array(shared_ref, copy=True)
        ref_tensor = (
            torch.from_numpy(ref_array)
            .permute(2, 0, 1)
            .unsqueeze(1)
            .to(dtype=torch.float32)
        )
        cropped = center_crop(ref_tensor, 480, 832)
        self.assertEqual(tuple(cropped.shape), (3, 1, 480, 832))
        expected = torch.tensor([17.0, 33.0, 65.0]).view(3, 1, 1, 1)
        self.assertTrue(torch.allclose(cropped, expected.expand_as(cropped)))

        # Preserve the shared behavior for models without an explicit custom
        # canvas contract.
        self.assertEqual(
            resolve_fit(
                {"fit_into_canvas_image_refs": 0},
                auto_aspect=False,
            ),
            1,
        )
        handler_source = _read(_WAN_HANDLER_PATH)
        self.assertIn(
            'extra_model_def["custom_image_ref_postprocessor_handles_canvas"] = True',
            handler_source,
        )

    def test_fake_identity_start_does_not_claim_canvas_fit(self):
        source = _read(_WGP_PATH)
        self.assertIn(
            "if sample_fit_canvas != None and not fake_start_image:",
            source,
        )

    def test_fake_identity_start_is_not_passed_as_first_window_history(self):
        source = _read(_WGP_PATH)
        self.assertIn(
            "input_video_for_model = None if fake_start_image and window_no == 1 else pre_video_guide",
            source,
        )

    def test_fake_identity_start_is_not_concatenated_into_output(self):
        source = _read(_WGP_PATH)
        self.assertIn(
            "if fake_start_image and window_no == 1:\n                    prefix_video = None",
            source,
        )
        self.assertNotIn(
            "prefix_video[:, :-source_video_overlap_frames_count] if fake_start_image",
            source,
        )

    def test_recast_preroll_shifts_timeline_and_uses_reverse_motion(self):
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest("torch is unavailable") from exc

        helpers = _load_functions(
            _WGP_PATH,
            (
                "_resolve_scail2_recast_warmup_frames",
                "_shift_guide_window_for_warmup",
                "_prepend_first_video_frame",
                "_prepend_reverse_motion_preroll",
            ),
            {"torch": torch},
        )
        resolve = helpers["_resolve_scail2_recast_warmup_frames"]
        shift = helpers["_shift_guide_window_for_warmup"]
        repeat_first = helpers["_prepend_first_video_frame"]
        prepend_motion = helpers["_prepend_reverse_motion_preroll"]

        self.assertEqual(
            resolve(
                {"scail2_recast_warmup_frames": 8},
                {"scail2": True},
                "V01AI",
                4,
            ),
            8,
        )
        self.assertEqual(shift(0, 81, 8), (-8, 73))
        self.assertEqual(shift(76, 157, 8), (68, 149))

        raw = torch.arange(10, dtype=torch.uint8).view(10, 1, 1, 1)
        padded = prepend_motion(raw, 8)
        self.assertEqual(tuple(padded.shape), (18, 1, 1, 1))
        self.assertEqual(
            padded[:9, 0, 0, 0].tolist(),
            [8, 7, 6, 5, 4, 3, 2, 1, 0],
        )
        self.assertEqual(padded[9:, 0, 0, 0].tolist(), list(range(1, 10)))

        short = torch.arange(3, dtype=torch.uint8).view(3, 1, 1, 1)
        padded = prepend_motion(short, 8)
        self.assertEqual(tuple(padded.shape), (11, 1, 1, 1))
        self.assertEqual(
            padded[:, 0, 0, 0].tolist(),
            [2, 2, 2, 2, 2, 2, 2, 1, 0, 1, 2],
        )

        repeated = repeat_first(short, 2)
        self.assertEqual(repeated[:, 0, 0, 0].tolist(), [0, 0, 0, 1, 2])
        self.assertIn(
            "prepend_padding = (\n"
            "        _prepend_reverse_motion_preroll",
            _read(_WGP_PATH),
        )

    def test_control_dimensions_are_shared_by_animate_and_replace(self):
        source = _read(_SCAIL2_PATH)
        tree = ast.parse(source, filename="app/models/wan/scail2/__init__.py")
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "custom_preprocess_scail2"
        )
        segment = ast.get_source_segment(source, function) or ""
        replace_branch = segment.index("if replace_mode:")
        self.assertLess(segment.index("source_h, source_w = _video_hw(video_guide)"), replace_branch)
        self.assertLess(segment.index("calculate_new_dimensions("), replace_branch)

    def test_identity_reference_is_center_cropped_into_video_canvas(self):
        source = _read(_SCAIL2_PATH)
        tree = ast.parse(source, filename="app/models/wan/scail2/__init__.py")
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_resize_ref_image_for_mode"
        )
        calls = {
            node.func.id for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("_center_crop_ref_image_to_canvas", calls)
        self.assertNotIn("calculate_new_dimensions", calls)

    def test_recast_preparation_keeps_the_complete_subject_in_wide_canvas(self):
        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_recast_subject_crop_box",
                "_fit_recast_reference_layers",
            ),
        )
        rgb = np.full((1000, 1000, 3), 127, dtype=np.uint8)
        # A tall subject whose face would be discarded by the old square ->
        # 16:9 fill-and-center-crop.
        rgb[80:180, 430:570] = (240, 30, 30)
        rgb[180:920, 360:640] = (20, 60, 210)
        semantic = np.zeros_like(rgb)
        semantic[80:920, 360:640] = (0, 0, 255)

        prepared, mask = helpers["_fit_recast_reference_layers"](
            rgb, semantic, 832, 480, aligned=False,
        )
        prepared_array = np.asarray(prepared)
        mask_array = np.asarray(mask)

        self.assertEqual(prepared.size, (832, 480))
        self.assertEqual(mask.size, (832, 480))
        self.assertGreater(int(prepared_array[..., 0].max()), 200)
        self.assertTrue(bool(np.all(mask_array == (0, 0, 255), axis=-1).any()))
        # Standalone references are contained on a neutral canvas, not filled
        # and cropped through the subject's head.
        self.assertTrue(bool(np.all(prepared_array == 127, axis=-1).any()))

    def test_isolation_keeps_original_rgb_for_clip_identity_only(self):
        import tempfile
        from PIL import Image

        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_recast_subject_crop_box",
                "_fit_recast_reference_layers",
                "_prepare_recast_reference_frame",
            ),
        )
        rgb = np.full((48, 80, 3), (12, 34, 56), dtype=np.uint8)
        rgb[8:44, 24:56] = (210, 80, 30)
        semantic = np.zeros_like(rgb)
        semantic[8:44, 24:56] = (0, 0, 255)

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = os.path.join(temp_dir, "reference.png")
            Image.fromarray(rgb).save(reference_path)
            prepared = helpers["_prepare_recast_reference_frame"](
                reference_path,
                (0, 0, 255),
                isolate_reference=True,
                aligned_semantic_mask=semantic,
                width=80,
                height=48,
            )

        spatial = np.asarray(prepared["image"])
        identity = np.asarray(prepared["identity_image"])
        self.assertTrue(np.all(identity[0, 0] == (12, 34, 56)))
        self.assertTrue(np.all(spatial[0, 0] == (127, 127, 127)))
        self.assertTrue(np.all(identity[20, 40] == spatial[20, 40]))

    def test_isolated_spatial_reference_uses_source_scene_not_gray(self):
        import tempfile
        from PIL import Image

        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_recast_subject_crop_box",
                "_fit_recast_reference_layers",
                "_prepare_recast_reference_frame",
            ),
        )
        rgb = np.full((64, 48, 3), (220, 220, 220), dtype=np.uint8)
        rgb[5:59, 15:33] = (35, 80, 190)
        subject = np.zeros((64, 48), dtype=bool)
        subject[5:59, 15:33] = True
        opacity = subject.astype(np.float32)
        scene = np.full((64, 48, 3), (18, 44, 73), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = os.path.join(temp_dir, "reference.png")
            Image.fromarray(rgb).save(reference_path)
            prepared = helpers["_prepare_recast_reference_frame"](
                reference_path,
                (0, 0, 255),
                isolate_reference=True,
                subject_conditioning=(subject, opacity, "test instance"),
                spatial_background=scene,
                width=48,
                height=64,
            )

        spatial = np.asarray(prepared["image"])
        semantic = np.any(np.asarray(prepared["mask"]) > 30, axis=-1)
        identity = np.asarray(prepared["identity_image"])
        self.assertTrue(bool(semantic.any()))
        self.assertTrue(np.all(spatial[~semantic] == (18, 44, 73)))
        self.assertFalse(bool(np.all(spatial[~semantic] == 127, axis=-1).any()))
        self.assertTrue(np.all(identity[0, 0] == (220, 220, 220)))

    def test_soft_reference_edge_composites_directly_onto_source_scene(self):
        import tempfile
        from PIL import Image

        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_recast_subject_crop_box",
                "_fit_recast_reference_layers",
                "_prepare_recast_reference_frame",
            ),
        )
        rgb = np.full((20, 20, 3), (220, 220, 220), dtype=np.uint8)
        subject = np.zeros((20, 20), dtype=bool)
        subject[2:18, 2:18] = True
        opacity = subject.astype(np.float32)
        opacity[2, 2:18] = 0.25
        scene = np.full((20, 20, 3), (20, 40, 80), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = os.path.join(temp_dir, "reference.png")
            Image.fromarray(rgb).save(reference_path)
            prepared = helpers["_prepare_recast_reference_frame"](
                reference_path,
                (0, 0, 255),
                isolate_reference=True,
                subject_conditioning=(
                    subject,
                    opacity,
                    "test soft edge",
                    rgb,
                ),
                spatial_background=scene,
                width=20,
                height=20,
            )

        spatial = np.asarray(prepared["image"])
        # 25% foreground + 75% source scene. The old ordering produced a
        # neutral-gray boundary here before replacing the scene background.
        self.assertTrue(np.allclose(
            spatial[2, 10],
            np.asarray((70, 85, 115)),
            atol=1,
        ))
        self.assertTrue(np.all(spatial[0, 0] == (20, 40, 80)))


class TestRuntimeCustomSettings(unittest.TestCase):
    def test_endpoint_owned_scail_settings_survive_strict_collection(self):
        helpers = _load_functions(
            _WGP_PATH,
            (
                "get_custom_setting_key",
                "_normalize_custom_setting_type",
                "_normalize_custom_setting_name",
                "get_custom_setting_id",
                "get_model_custom_settings",
                "parse_custom_setting_typed_value",
                "collect_custom_settings_from_inputs",
            ),
            {
                "re": re,
                "CUSTOM_SETTINGS_MAX": 16,
                "CUSTOM_SETTING_TYPES": {"text", "int", "float", "dropdown"},
            },
        )
        collect = helpers["collect_custom_settings_from_inputs"]
        model_def = {
            "custom_settings": [{
                "id": "image_ref_keyword_content",
                "type": "text",
                "label": "Reference keyword",
            }],
            "runtime_custom_settings": [
                "scail2_reference_mask_path",
                "scail2_additional_reference_mask_paths",
                "scail2_reference_expected_colors",
                "scail2_clip_reference_path",
                "scail2_identity_latent_isolation",
                "scail2_identity_latent_reference_index",
                "scail2_recast_warmup_frames",
                "scail2_primary_only_continuations",
            ],
        }
        parsed, error = collect(
            model_def,
            {
                "custom_settings": {
                    "image_ref_keyword_content": "person",
                    "scail2_reference_mask_path": "primary.png",
                    "scail2_additional_reference_mask_paths": ["view.png"],
                    "scail2_reference_expected_colors": [[255, 0, 0]],
                    "scail2_clip_reference_path": "identity.png",
                    "scail2_identity_latent_isolation": True,
                    "scail2_identity_latent_reference_index": 1,
                    "scail2_recast_warmup_frames": 8,
                    "scail2_primary_only_continuations": True,
                    "untrusted_unknown_key": "discard me",
                },
            },
            strict=True,
        )

        self.assertIsNone(error)
        self.assertEqual(parsed["scail2_reference_mask_path"], "primary.png")
        self.assertEqual(
            parsed["scail2_additional_reference_mask_paths"], ["view.png"],
        )
        self.assertEqual(
            parsed["scail2_reference_expected_colors"], [[255, 0, 0]],
        )
        self.assertEqual(
            parsed["scail2_clip_reference_path"], "identity.png",
        )
        self.assertIs(parsed["scail2_identity_latent_isolation"], True)
        self.assertEqual(parsed["scail2_identity_latent_reference_index"], 1)
        self.assertEqual(parsed["scail2_recast_warmup_frames"], 8)
        self.assertIs(parsed["scail2_primary_only_continuations"], True)
        self.assertNotIn("untrusted_unknown_key", parsed)


class TestScail2ReferenceIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import torch.nn.functional as functional
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise unittest.SkipTest("torch or Pillow is unavailable") from exc

        def to_rgb_tensor(color, device="cpu", dtype=None):
            return torch.as_tensor(color, device=device, dtype=dtype)

        def convert_image_to_tensor(image):
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
            return (
                torch.from_numpy(pixels.copy())
                .permute(2, 0, 1)
                .div(127.5)
                .sub(1.0)
            )

        cls.torch = torch
        cls.image = Image
        cls.helpers = _load_functions(
            _SCAIL2_PATH,
            (
                "_resize_ref_image",
                "_center_crop_ref_image_to_canvas",
                "_center_crop_ref_alpha_to_canvas",
                "_load_reference_alpha",
                "_load_clip_identity_reference",
                "_get_scail2_recast_warmup_frames",
                "_use_scail2_primary_only_continuation_refs",
                "_semantic_reference_alpha",
                "_blend_scail2_identity_latents",
                "_isolate_scail2_reference",
                "_build_dynamic_source_scene_reference",
                "_active_scail2_reference_colors",
                "_select_scail2_window_reference_indices",
                "normalize_single_color_mask",
            ),
            {
                "Path": __import__("pathlib").Path,
                "Image": Image,
                "ImageOps": ImageOps,
                "np": np,
                "logging": __import__("logging"),
                "torch": torch,
                "F": functional,
                "to_rgb_tensor": to_rgb_tensor,
                "convert_image_to_tensor": convert_image_to_tensor,
            },
        )

    def test_original_png_alpha_is_recovered_at_preprocessed_size(self):
        import tempfile

        load_alpha = self.helpers["_load_reference_alpha"]
        rgba = np.zeros((8, 12, 4), dtype=np.uint8)
        rgba[..., :3] = (20, 40, 60)
        rgba[2:7, 3:10, 3] = 255
        with tempfile.TemporaryDirectory() as temp_dir:
            alpha_path = os.path.join(temp_dir, "cutout.png")
            self.image.fromarray(rgba).save(alpha_path)
            alpha = load_alpha(
                {"scail2_reference_alpha_path": alpha_path},
                4, 6, "cpu", self.torch.float32,
            )

        self.assertIsNotNone(alpha)
        self.assertEqual(tuple(alpha.shape), (1, 1, 4, 6))
        self.assertGreater(float(alpha.max()), 0.9)
        self.assertLess(float(alpha.min()), 0.1)

    def test_semantic_mask_neutralizes_unrelated_reference_pixels(self):
        torch = self.torch
        isolate = self.helpers["_isolate_scail2_reference"]
        image = torch.full((3, 1, 4, 6), -0.8, dtype=torch.float32)
        image[:, :, 1:3, 2:4] = 0.6
        mask = torch.full_like(image, -1.0)
        mask[2:3, :, 1:3, 2:4] = 1.0

        isolated = isolate(
            image, mask, {"ref_matte_background_color": [127, 127, 127]},
        )

        neutral = 127.0 / 127.5 - 1.0
        self.assertTrue(torch.allclose(isolated[:, :, 0, 0], torch.full((3, 1), neutral)))
        self.assertTrue(torch.equal(isolated[:, :, 1:3, 2:4], image[:, :, 1:3, 2:4]))

    def test_clip_identity_reference_keeps_original_rgb_separate(self):
        import tempfile

        load_identity = self.helpers["_load_clip_identity_reference"]
        pixels = np.zeros((4, 6, 3), dtype=np.uint8)
        pixels[:] = (12, 34, 56)
        pixels[1:3, 2:5] = (210, 80, 30)
        with tempfile.TemporaryDirectory() as temp_dir:
            identity_path = os.path.join(temp_dir, "identity.png")
            self.image.fromarray(pixels).save(identity_path)
            identity = load_identity(
                {"scail2_clip_reference_path": identity_path},
                4,
                6,
                "cpu",
                self.torch.float32,
            )

        self.assertEqual(tuple(identity.shape), (3, 1, 4, 6))
        expected_background = (
            self.torch.tensor([12.0, 34.0, 56.0]).div(127.5).sub(1.0)
        )
        self.assertTrue(
            self.torch.allclose(identity[:, 0, 0, 0], expected_background),
        )

    def test_identity_latent_isolation_keeps_original_only_on_subject(self):
        torch = self.torch
        blend = self.helpers["_blend_scail2_identity_latents"]
        identity = torch.ones((2, 1, 4, 5), dtype=torch.float32)
        isolated = torch.zeros_like(identity)
        ref_mask = torch.full((3, 1, 4, 5), -1.0, dtype=torch.float32)
        ref_mask[2, 0, 1:3, 2:4] = 1.0

        blended = blend(identity, isolated, ref_mask, margin=0)

        self.assertTrue(
            torch.equal(
                blended[:, :, 1:3, 2:4],
                torch.ones((2, 1, 2, 2)),
            )
        )
        self.assertEqual(float(blended[:, :, 0, 0].max()), 0.0)
        self.assertEqual(float(blended[:, :, 3, 4].max()), 0.0)

    def test_recast_warmup_setting_is_bounded_and_tolerates_bad_metadata(self):
        get_warmup = self.helpers["_get_scail2_recast_warmup_frames"]
        self.assertEqual(get_warmup({"scail2_recast_warmup_frames": 8}), 8)
        self.assertEqual(get_warmup({"scail2_recast_warmup_frames": 200}), 32)
        self.assertEqual(get_warmup({"scail2_recast_warmup_frames": "bad"}), 0)

    def test_recast_supporting_refs_are_limited_to_the_first_window(self):
        use_primary_only = self.helpers[
            "_use_scail2_primary_only_continuation_refs"
        ]
        enabled = {"scail2_primary_only_continuations": True}
        self.assertFalse(use_primary_only(enabled, 0))
        self.assertTrue(use_primary_only(enabled, 5))
        self.assertFalse(use_primary_only({}, 5))
        self.assertFalse(use_primary_only(enabled, "bad"))

    def test_source_alpha_preserves_soft_cutout_edges(self):
        torch = self.torch
        isolate = self.helpers["_isolate_scail2_reference"]
        image = torch.ones((3, 1, 2, 3), dtype=torch.float32)
        mask = torch.full_like(image, -1.0)
        alpha = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
        alpha[:, :, 0, 1] = 0.5
        alpha[:, :, 1, 1] = 1.0

        isolated = isolate(
            image, mask, {"ref_matte_background_color": [127, 127, 127]},
            alpha_mask=alpha,
        )

        neutral = 127.0 / 127.5 - 1.0
        self.assertAlmostEqual(float(isolated[0, 0, 0, 0]), neutral, places=5)
        self.assertAlmostEqual(float(isolated[0, 0, 0, 1]), (1.0 + neutral) / 2.0, places=5)
        self.assertEqual(float(isolated[0, 0, 1, 1]), 1.0)

    def test_explicit_additional_reference_keeps_its_assigned_color(self):
        torch = self.torch
        normalize = self.helpers["normalize_single_color_mask"]
        model_def = {
            "magic_mask_object_colors": [
                (0, 0, 255),
                (255, 0, 0),
                (0, 255, 0),
            ],
        }
        red_mask = torch.full((3, 1, 4, 6), -1.0, dtype=torch.float32)
        red_mask[0, :, 1:3, 2:5] = 1.0

        kept = normalize(red_mask, model_def, (255, 0, 0))
        remapped = normalize(red_mask, model_def, (0, 0, 255))

        self.assertTrue(torch.equal(kept, red_mask))
        self.assertGreater(float(remapped[2, :, 1:3, 2:5].min()), 0.9)
        self.assertLess(float(remapped[0, :, 1:3, 2:5].max()), -0.9)

    def test_source_scene_binary_visibility_mask_is_not_recolored(self):
        torch = self.torch
        normalize = self.helpers["normalize_single_color_mask"]
        model_def = {
            "magic_mask_object_colors": [
                (0, 0, 255),
                (255, 0, 0),
            ],
        }
        scene_mask = torch.ones((3, 1, 8, 12), dtype=torch.float32)
        scene_mask[:, :, 2:7, 4:9] = -1.0

        normalized = normalize(scene_mask, model_def, None)

        self.assertTrue(torch.equal(normalized, scene_mask))

    def test_dynamic_scene_reference_refreshes_from_requested_window_frame(self):
        torch = self.torch
        build_scene = self.helpers[
            "_build_dynamic_source_scene_reference"
        ]
        frames = torch.zeros((3, 4, 16, 20), dtype=torch.float32)
        frames[:, 0] = -0.75
        frames[:, 1] = -0.25
        frames[:, 2] = 0.50
        frames[:, 3] = 0.80
        masks = torch.ones_like(frames)
        masks[0:2, 2, 5:11, 7:13] = -1.0

        scene = build_scene(
            frames,
            masks,
            2,
            {"ref_matte_background_color": [127, 127, 127]},
        )

        neutral = 127.0 / 127.5 - 1.0
        self.assertEqual(scene["frame_index"], 2)
        self.assertEqual(scene["expansion_pixels"], 2)
        self.assertAlmostEqual(
            float(scene["image"][0, 0, 0, 0]),
            0.50,
            places=5,
        )
        self.assertAlmostEqual(
            float(scene["image"][0, 0, 7, 9]),
            neutral,
            places=5,
        )
        self.assertEqual(float(scene["mask"][0, 0, 0, 0]), 1.0)
        self.assertEqual(float(scene["mask"][0, 0, 7, 9]), -1.0)
        self.assertGreater(scene["hidden_fraction"], 0.05)

    def test_timeline_scene_reference_inpaints_instead_of_adding_gray(self):
        torch = self.torch
        build_scene = self.helpers[
            "_build_dynamic_source_scene_reference"
        ]
        frames = torch.full(
            (3, 2, 24, 32),
            0.55,
            dtype=torch.float32,
        )
        frames[:, 1, 7:18, 11:22] = -0.85
        masks = torch.ones_like(frames)
        masks[0:2, 1, 7:18, 11:22] = -1.0

        scene = build_scene(
            frames,
            masks,
            1,
            {"ref_matte_background_color": [127, 127, 127]},
            inpaint_hidden=True,
        )

        neutral = 127.0 / 127.5 - 1.0
        center = float(scene["image"][0, 0, 12, 16])
        self.assertGreater(center, 0.35)
        self.assertGreater(abs(center - neutral), 0.25)
        self.assertEqual(float(scene["mask"][0, 0, 12, 16]), -1.0)

    def test_dynamic_scene_reference_uses_nearest_frame_with_a_target(self):
        torch = self.torch
        build_scene = self.helpers[
            "_build_dynamic_source_scene_reference"
        ]
        frames = torch.zeros((3, 4, 12, 16), dtype=torch.float32)
        frames[:, 1] = 0.25
        frames[:, 2] = 0.50
        masks = torch.ones_like(frames)
        masks[0:2, 2, 4:8, 6:10] = -1.0

        scene = build_scene(frames, masks, 1, {})

        self.assertIsNotNone(scene)
        self.assertEqual(scene["frame_index"], 2)
        self.assertAlmostEqual(
            float(scene["image"][0, 0, 0, 0]),
            0.50,
            places=5,
        )

    def test_multi_character_views_are_routed_by_active_window_colors(self):
        torch = self.torch
        detect = self.helpers["_active_scail2_reference_colors"]
        select = self.helpers[
            "_select_scail2_window_reference_indices"
        ]
        blue = (0, 0, 255)
        red = (255, 0, 0)
        mask = torch.ones((3, 4, 40, 60), dtype=torch.float32)
        mask[0:2, 1:3, 5:20, 5:20] = -1.0
        mask[1:3, 2:4, 18:35, 35:55] = -1.0

        active = detect(mask, [None, blue, blue, red, red])

        self.assertEqual(active, [blue, red])
        self.assertEqual(
            select([None, blue, blue, red, red], [blue]),
            [1, 2],
        )
        self.assertEqual(
            select([None, blue, blue, red, red], [red]),
            [3, 4],
        )
        self.assertEqual(
            select([None, blue, blue, red, red], [blue, red]),
            [0],
        )
        self.assertEqual(
            select([None, blue, blue, red, red], []),
            [0],
        )
        self.assertEqual(
            select([blue, blue, red, red], [red]),
            [2, 3],
        )


class TestMultiPersonRecast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_normalize_recast_person_count",
                "_normalize_recast_resolution_profile",
                "_normalize_scail2_inference_steps",
                "_normalize_scail2_guidance_scale",
                "_recast_window_size_for_profile",
                "_build_recast_prompt",
                "_extract_output_seed",
                "_count_recast_mask_people",
                "_align_recast_reference_mask",
                "_recast_color_region",
                "_matte_recast_reference_frame",
                "_recast_reference_canvas_size",
                "_recast_resolution_for_source",
                "_build_recast_source_scene_layers",
                "_build_recast_source_scene_reference",
                "_enable_recast_dynamic_source_scene_reference",
                "_compose_recast_group_reference_frame",
                "_compose_recast_cast_reference_frame",
                "_compose_recast_character_masks",
                "_detect_recast_shot_ranges",
                "_summarize_recast_mapping_mask",
                "_select_recast_timeline_anchor",
                "_normalize_recast_tracking_target",
                "_recast_mask_region",
                "_find_recast_unmapped_shots",
                "_recast_appearance_descriptor",
                "_build_recast_mapping_appearance_prototypes",
                "_recover_recast_unmapped_shots",
                "_plan_recast_shot_segments",
                "_resample_recast_tracking_timeline",
                "_remap_recast_shot_mask",
                "_quantize_recast_shot_frame_count",
                "_build_recast_shot_prompt",
                "_compose_recast_native_people_masks",
                "_select_recast_generated_regions",
                "_normalize_repaint_region_mappings",
                "_repaint_resolution_for_aspect",
                "_validate_repaint_target_aspect",
            ),
            {
                "_RECAST_MASK_COLORS": [
                    (0, 0, 255), (255, 0, 0), (0, 255, 0),
                    (255, 0, 255), (0, 255, 255),
                ],
                "_RECAST_PROTECTION_COLORS": [
                    (0, 0, 255), (255, 0, 0), (0, 255, 0),
                    (255, 0, 255), (0, 255, 255), (255, 255, 0),
                ],
                "_RECAST_RESOLUTION_PROFILES": {
                    "480p": (832, 480),
                    "512p": (896, 512),
                    "704p": (1280, 704),
                },
            },
        )

    def test_person_count_is_clamped_to_scail2_range(self):
        normalize = self.helpers["_normalize_recast_person_count"]
        self.assertEqual(normalize(None), 1)
        self.assertEqual(normalize("bad"), 1)
        self.assertEqual(normalize(-2), 1)
        self.assertEqual(normalize(3), 3)
        self.assertEqual(normalize(99), 5)

    def test_independent_characters_form_one_native_color_mapped_cast(self):
        compose = self.helpers["_compose_recast_cast_reference_frame"]
        blue = np.asarray((0, 0, 255), dtype=np.uint8)
        red = np.asarray((255, 0, 0), dtype=np.uint8)
        first_rgb = np.zeros((80, 60, 3), dtype=np.uint8)
        first_rgb[10:75, 15:45] = (220, 200, 180)
        first_mask = np.zeros_like(first_rgb)
        first_mask[10:75, 15:45] = blue
        second_rgb = np.zeros((80, 60, 3), dtype=np.uint8)
        second_rgb[5:75, 12:48] = (80, 40, 30)
        second_mask = np.zeros_like(second_rgb)
        second_mask[5:75, 12:48] = red

        cast, cast_mask = compose(
            [
                (first_rgb, first_mask),
                (second_rgb, second_mask),
            ],
            2,
            (240, 120),
        )

        self.assertEqual(cast.shape, (120, 240, 3))
        self.assertEqual(cast_mask.shape, cast.shape)
        blue_region = np.all(cast_mask == blue, axis=-1)
        red_region = np.all(cast_mask == red, axis=-1)
        self.assertTrue(bool(blue_region.any()))
        self.assertTrue(bool(red_region.any()))
        self.assertFalse(bool(np.logical_and(blue_region, red_region).any()))
        self.assertGreater(float(cast[blue_region].mean()), 100.0)
        self.assertLess(float(cast[red_region].mean()), 100.0)

    def test_colorized_preview_counts_distinct_people(self):
        count_people = self.helpers["_count_recast_mask_people"]
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
        mask = np.zeros((4, 4, 3), dtype=np.uint8)
        mask[0, 0] = colors[0]
        mask[1, 1] = colors[1]
        self.assertEqual(count_people(mask, colors), 2)

    def test_recast_reference_canvas_follows_control_orientation(self):
        canvas_size = self.helpers["_recast_reference_canvas_size"]

        self.assertEqual(
            canvas_size(np.zeros((72, 128, 3), dtype=np.uint8)),
            (832, 480),
        )
        self.assertEqual(
            canvas_size(np.zeros((292, 216, 3), dtype=np.uint8)),
            (480, 640),
        )
        self.assertEqual(
            canvas_size(np.zeros((80, 80, 3), dtype=np.uint8)),
            (480, 480),
        )

    def test_recast_quality_profiles_raise_only_the_spatial_budget(self):
        normalize = self.helpers["_normalize_recast_resolution_profile"]
        resolution = self.helpers["_recast_resolution_for_source"]
        landscape = np.zeros((72, 128, 3), dtype=np.uint8)
        portrait = np.zeros((292, 216, 3), dtype=np.uint8)

        self.assertEqual(normalize(None), "480p")
        self.assertEqual(normalize("896x512"), "512p")
        self.assertEqual(normalize("1280x704"), "704p")
        self.assertEqual(normalize("unsupported"), "480p")
        self.assertEqual(resolution(landscape, "480p"), (832, 480))
        self.assertEqual(resolution(landscape, "512p"), (896, 512))
        self.assertEqual(resolution(landscape, "704p"), (1248, 704))
        self.assertEqual(resolution(portrait, "512p"), (512, 672))
        self.assertEqual(resolution(portrait, "704p"), (704, 928))
        for size in (
            resolution(landscape, "512p"),
            resolution(portrait, "512p"),
            resolution(landscape, "704p"),
            resolution(portrait, "704p"),
        ):
            self.assertEqual(size[0] % 32, 0)
            self.assertEqual(size[1] % 32, 0)

    def test_recast_704_profile_adapts_window_to_vram(self):
        window = self.helpers["_recast_window_size_for_profile"]

        self.assertEqual(window("480p", 12), 81)
        self.assertEqual(window("512p", 12), 81)
        self.assertEqual(window("704p", None), 41)
        self.assertEqual(window("704p", 0), 41)
        self.assertEqual(window("704p", 12), 0)
        self.assertEqual(window("704p", 15.9), 0)
        self.assertEqual(window("704p", 16), 33)
        self.assertEqual(window("704p", 19.9), 33)
        self.assertEqual(window("704p", 20), 41)
        self.assertEqual(window("704p", 23.9), 41)
        self.assertEqual(window("704p", 24), 49)
        self.assertEqual(window("704p", 48), 49)
        for frames in (33, 41, 49, 81):
            self.assertEqual((frames - 1) % 4, 0)

    def test_scail_edit_sampling_values_are_clamped_and_user_adjustable(self):
        steps = self.helpers["_normalize_scail2_inference_steps"]
        guidance = self.helpers["_normalize_scail2_guidance_scale"]

        self.assertEqual(steps(None, 8), 8)
        self.assertEqual(steps(6, 8), 6)
        self.assertEqual(steps("12", 8), 12)
        self.assertEqual(steps(0, 8), 1)
        self.assertEqual(steps(99, 8), 50)
        self.assertEqual(steps("bad", 40), 40)
        self.assertEqual(guidance(None, 5.0), 5.0)
        self.assertEqual(guidance(3.5, 5.0), 3.5)
        self.assertEqual(guidance(-1, 5.0), 0.0)
        self.assertEqual(guidance(99, 5.0), 20.0)
        self.assertEqual(guidance("nan", 5.0), 5.0)

    def test_reference_subject_selection_keeps_one_sam3_instance(self):
        colors = [
            (0, 0, 255), (255, 0, 0), (0, 255, 0),
            (255, 0, 255), (0, 255, 255),
        ]
        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_clean_recast_reference_region",
                "_select_recast_reference_instance",
            ),
        )
        mask = np.zeros((120, 160, 3), dtype=np.uint8)
        mask[12:112, 48:112] = colors[0]
        mask[18:60, 5:37] = colors[1]
        mask[0:8, 145:158] = colors[0]

        selected, metadata = helpers[
            "_select_recast_reference_instance"
        ](mask, colors)

        self.assertTrue(bool(selected[50, 80]))
        self.assertFalse(bool(selected[30, 20]))
        self.assertFalse(bool(selected[3, 150]))
        self.assertEqual(metadata["instance_count"], 2)
        self.assertEqual(metadata["discarded_components"], 1)

    def test_reference_subject_mask_requests_colored_sam3_instances(self):
        import sys
        import types
        from unittest.mock import patch

        colors = [
            (0, 0, 255), (255, 0, 0), (0, 255, 0),
            (255, 0, 255), (0, 255, 255),
        ]
        mask = np.zeros((80, 120, 3), dtype=np.uint8)
        mask[8:76, 42:82] = colors[0]
        mask[15:48, 4:28] = colors[1]
        calls = []

        def fake_generate(*_args, **kwargs):
            calls.append(kwargs)
            return mask[None]

        fake_shared = types.ModuleType("shared")
        fake_shared.magic_mask = types.SimpleNamespace(
            generate_keyword_masks=fake_generate,
        )
        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_clean_recast_reference_region",
                "_select_recast_reference_instance",
                "_recast_reference_subject_mask",
            ),
        )

        with patch.dict(sys.modules, {"shared": fake_shared}):
            selected, opacity, source = helpers[
                "_recast_reference_subject_mask"
            ](
                np.zeros((80, 120, 3), dtype=np.uint8),
                np.full((80, 120), 255, dtype=np.uint8),
            )

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["colorize_objects"], True)
        self.assertEqual(calls[0]["max_colored_objects"], 5)
        self.assertTrue(bool(selected[40, 60]))
        self.assertFalse(bool(selected[25, 15]))
        self.assertTrue(np.array_equal(opacity, selected.astype(np.float32)))
        self.assertIn("instance", source)

    def test_u2net_matte_is_limited_to_the_sam3_selected_person(self):
        constrain = _load_functions(
            _LAUNCH_PATH,
            ("_constrain_recast_u2net_matte",),
        )["_constrain_recast_u2net_matte"]
        selected = np.zeros((30, 50), dtype=bool)
        selected[5:25, 5:20] = True
        opacity = selected.astype(np.float32)
        u2net_alpha = np.zeros((30, 50), dtype=np.float32)
        u2net_alpha[5:25, 5:20] = 1.0
        u2net_alpha[5, 5:20] = 0.2
        # U2Net may see another person, but SAM3 did not select this one.
        u2net_alpha[5:25, 32:47] = 1.0

        refined = constrain(selected, opacity, u2net_alpha)

        self.assertGreater(float(refined[15, 12]), 0.95)
        self.assertGreater(float(refined[5, 12]), 0.0)
        self.assertLess(float(refined[5, 12]), 0.95)
        self.assertEqual(float(refined[15, 39]), 0.0)

    def test_reference_refinement_uses_u2net_color_inside_sam3_gate(self):
        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_decontaminate_recast_reference_edges",
                "_constrain_recast_u2net_matte",
                "_refine_recast_reference_cutout",
            ),
        )
        frame = np.full((30, 50, 3), 245, dtype=np.uint8)
        selected = np.zeros((30, 50), dtype=bool)
        selected[5:25, 5:20] = True
        frame[7:23, 7:18] = (20, 60, 180)
        opacity = selected.astype(np.float32)

        def fake_matte(_frame, *, reference_path=None):
            self.assertEqual(reference_path, "reference.jpg")
            foreground = np.full_like(_frame, (18, 58, 176))
            alpha = np.zeros(selected.shape, dtype=np.float32)
            alpha[5:25, 5:20] = 1.0
            alpha[5:25, 32:47] = 1.0
            return foreground, alpha

        subject, refined_alpha, source, foreground = helpers[
            "_refine_recast_reference_cutout"
        ](
            frame,
            selected,
            opacity,
            "person instance",
            reference_path="reference.jpg",
            matte_runner=fake_matte,
        )

        self.assertIn("U2Net soft edge", source)
        self.assertTrue(bool(subject[15, 12]))
        self.assertFalse(bool(subject[15, 39]))
        self.assertGreater(float(refined_alpha[15, 12]), 0.95)
        self.assertEqual(float(refined_alpha[15, 39]), 0.0)
        self.assertTrue(np.all(foreground[5, 12] == (18, 58, 176)))

    def test_reference_refinement_falls_back_to_local_edge_cleanup(self):
        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_decontaminate_recast_reference_edges",
                "_constrain_recast_u2net_matte",
                "_refine_recast_reference_cutout",
            ),
        )
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        selected = np.zeros((24, 24), dtype=bool)
        selected[4:20, 4:20] = True
        frame[selected] = (250, 250, 250)
        frame[5:19, 5:19] = (20, 70, 180)

        def unavailable(*_args, **_kwargs):
            raise RuntimeError("offline")

        subject, opacity, source, foreground = helpers[
            "_refine_recast_reference_cutout"
        ](
            frame,
            selected,
            selected.astype(np.float32),
            "person instance",
            matte_runner=unavailable,
        )

        self.assertIn("local edge cleanup", source)
        self.assertTrue(np.array_equal(subject, selected))
        self.assertGreater(float(opacity[10, 10]), 0.95)
        self.assertLess(float(opacity[4, 10]), 1.0)
        self.assertLess(int(foreground[4, 10, 0]), 250)

    def test_reference_refinement_respects_authored_png_alpha(self):
        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_decontaminate_recast_reference_edges",
                "_constrain_recast_u2net_matte",
                "_refine_recast_reference_cutout",
            ),
        )
        frame = np.full((10, 10, 3), 80, dtype=np.uint8)
        selected = np.zeros((10, 10), dtype=bool)
        selected[2:8, 2:8] = True
        opacity = selected.astype(np.float32) * 0.75

        def must_not_run(*_args, **_kwargs):
            self.fail("U2Net should not replace an authored PNG alpha")

        result = helpers["_refine_recast_reference_cutout"](
            frame,
            selected,
            opacity,
            "png alpha",
            matte_runner=must_not_run,
        )

        self.assertTrue(np.array_equal(result[0], selected))
        self.assertTrue(np.array_equal(result[1], opacity))
        self.assertEqual(result[2], "png alpha")
        self.assertTrue(np.array_equal(result[3], frame))

    def test_character_mapping_masks_have_stable_card_colors(self):
        compose = self.helpers["_compose_recast_character_masks"]
        colors = [(0, 0, 255), (255, 0, 0)]
        first = np.zeros((2, 8, 12, 3), dtype=np.uint8)
        second = np.zeros_like(first)
        first[:, 1:7, 1:4] = colors[0]
        second[:, 1:7, 8:11] = colors[1]

        merged, overlaps = compose([first, second], colors)

        self.assertTrue(np.all(merged[:, 1:7, 1:4] == colors[0]))
        self.assertTrue(np.all(merged[:, 1:7, 8:11] == colors[1]))
        self.assertTrue(np.all(merged[:, 0, 0] == 255))
        self.assertEqual(overlaps, [0.0, 0.0])

    def test_character_mapping_rejects_ambiguous_overlap(self):
        compose = self.helpers["_compose_recast_character_masks"]
        first = np.zeros((1, 8, 12, 3), dtype=np.uint8)
        second = np.zeros_like(first)
        first[:, 1:7, 2:8] = (0, 0, 255)
        second[:, 1:7, 3:9] = (255, 0, 0)

        with self.assertRaisesRegex(ValueError, "overlaps an earlier mapping"):
            compose([first, second], [(0, 0, 255), (255, 0, 0)])

    def test_timeline_tracking_detects_hard_cuts(self):
        detect = self.helpers["_detect_recast_shot_ranges"]
        frames = np.empty((18, 36, 64, 3), dtype=np.uint8)
        frames[:8] = (12, 28, 44)
        frames[8:] = (220, 170, 70)

        self.assertEqual(detect(frames), [(0, 8), (8, 18)])

    def test_timeline_tracking_detects_same_location_character_cuts(self):
        detect = self.helpers["_detect_recast_shot_ranges"]
        first_character = np.full(
            (36, 64, 3),
            (70, 55, 40),
            dtype=np.uint8,
        )
        second_character = first_character.copy()
        first_character[4:34, 8:28] = (165, 145, 115)
        second_character[4:34, 36:56] = (35, 30, 25)
        frames = np.stack(
            [first_character] * 9 + [second_character] * 11,
        )

        # The shared background keeps this cut below the old 0.30 floor,
        # matching shot/reverse-shot footage where each actor appears alone.
        self.assertEqual(
            detect(frames, absolute_cut_threshold=0.30),
            [(0, 20)],
        )
        self.assertEqual(detect(frames), [(0, 9), (9, 20)])

    def test_timeline_tracking_ignores_ordinary_subject_motion(self):
        detect = self.helpers["_detect_recast_shot_ranges"]
        frames = []
        for offset in range(24):
            frame = np.full(
                (36, 64, 3),
                (70, 55, 40),
                dtype=np.uint8,
            )
            frame[6:32, 4 + offset:20 + offset] = (165, 145, 115)
            frames.append(frame)

        self.assertEqual(
            detect(np.stack(frames)),
            [(0, len(frames))],
        )

    def test_tracking_normalizes_wearing_to_sam_friendly_in_wording(self):
        normalize = self.helpers["_normalize_recast_tracking_target"]

        self.assertEqual(
            normalize("  Man   wearing black  "),
            "Man in black",
        )
        self.assertEqual(
            normalize("woman in a silver dress"),
            "woman in a silver dress",
        )

    def test_blank_shot_person_is_reacquired_by_source_appearance(self):
        find_unmapped = self.helpers["_find_recast_unmapped_shots"]
        recover = self.helpers["_recover_recast_unmapped_shots"]
        colors = [
            (0, 0, 255), (255, 0, 0), (0, 255, 0),
            (255, 0, 255), (0, 255, 255),
        ]
        frames = np.full((12, 48, 64, 3), (90, 75, 60), dtype=np.uint8)
        white_region = np.zeros((48, 64), dtype=bool)
        black_region = np.zeros_like(white_region)
        white_region[5:44, 8:27] = True
        black_region[5:44, 36:55] = True
        frames[0:4, white_region] = (225, 220, 205)
        frames[4:8, black_region] = (25, 30, 38)
        frames[8:12, black_region] = (28, 32, 40)

        blue = np.zeros_like(frames)
        red = np.zeros_like(frames)
        blue[0:4, white_region] = colors[0]
        red[4:8, black_region] = colors[1]
        shots = [(0, 4), (4, 8), (8, 12)]
        unresolved = find_unmapped([blue, red], shots)

        self.assertEqual(
            unresolved,
            [{
                "shot_index": 2,
                "start_frame": 8,
                "end_frame": 12,
            }],
        )

        generic = np.zeros((4, 48, 64, 3), dtype=np.uint8)
        generic[:, black_region] = colors[0]
        recovered_masks, recovered = recover(
            frames,
            [blue, red],
            shots,
            unresolved,
            generic,
            [(0, 4)],
        )

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["shot_index"], 2)
        self.assertEqual(recovered[0]["mapping_index"], 1)
        self.assertTrue(np.all(
            recovered_masks[1][8:12, black_region] == colors[1],
        ))
        self.assertFalse(bool(
            np.any(recovered_masks[0][8:12] > 30),
        ))

    def test_shot_plan_does_not_union_characters_across_camera_cuts(self):
        plan = self.helpers["_plan_recast_shot_segments"]
        blue = np.zeros((20, 12, 20, 3), dtype=np.uint8)
        red = np.zeros_like(blue)
        blue[0:6, 2:10, 2:6] = (0, 0, 255)
        red[6:12, 2:10, 14:18] = (255, 0, 0)
        blue[12:20, 2:10, 2:6] = (0, 0, 255)
        red[12:20, 2:10, 14:18] = (255, 0, 0)

        shots = plan(
            [blue, red],
            [(0, 6), (6, 12), (12, 20)],
        )

        self.assertEqual(
            [shot["active_mapping_indices"] for shot in shots],
            [[0], [1], [0, 1]],
        )
        self.assertEqual(
            [shot["mode"] for shot in shots],
            ["solo", "solo", "group"],
        )
        self.assertTrue(shots[2]["cooccurring"])
        self.assertIn(shots[2]["anchor_frame_index"], range(12, 20))

    def test_shot_mask_reassigns_active_character_to_local_blue(self):
        remap = self.helpers["_remap_recast_shot_mask"]
        semantic = np.full((2, 8, 12, 3), 255, dtype=np.uint8)
        semantic[:, 1:7, 1:4] = (0, 0, 255)
        semantic[:, 1:7, 8:11] = (255, 0, 0)

        local = remap(semantic, [1])

        self.assertTrue(np.all(local[:, 1:7, 8:11] == (0, 0, 255)))
        self.assertTrue(np.all(local[:, 1:7, 1:4] == 255))
        self.assertFalse(bool(np.all(local == (255, 0, 0), axis=-1).any()))

    def test_shot_timeline_resampling_keeps_contiguous_exact_ranges(self):
        resample = self.helpers["_resample_recast_tracking_timeline"]
        source = np.arange(12, dtype=np.uint8).reshape(12, 1, 1, 1)
        source = np.repeat(source, 3, axis=-1)
        mask = np.zeros_like(source)
        mask[:4] = (0, 0, 255)

        resized_source, resized_masks, ranges = resample(
            source,
            [mask],
            [(0, 4), (4, 12)],
            6,
        )

        self.assertEqual(len(resized_source), 6)
        self.assertEqual(len(resized_masks[0]), 6)
        self.assertEqual(ranges, [(0, 2), (2, 6)])
        self.assertEqual(ranges[-1][1], 6)

    def test_shot_lengths_are_padded_then_trimmed_to_exact_timeline(self):
        quantize = self.helpers["_quantize_recast_shot_frame_count"]

        self.assertEqual(quantize(85, 5, 4), (85, 0))
        self.assertEqual(quantize(86, 5, 4), (89, 3))
        self.assertEqual(quantize(2, 5, 4), (5, 3))

    def test_shot_prompt_contains_no_global_character_instruction(self):
        prompt = self.helpers["_build_recast_shot_prompt"]

        solo = prompt(1)
        group = prompt(2)
        full_cast = prompt(
            2,
            finished_video_prompt=(
                "A blonde woman in silver fights a redhead in black."
            ),
            total_mapping_count=2,
        )
        subset = prompt(
            2,
            finished_video_prompt="Three distinct characters fight.",
            total_mapping_count=3,
        )

        self.assertIn("A clearly defined character", solo)
        self.assertIn("2 clearly defined characters", group)
        self.assertEqual(
            full_cast,
            "A blonde woman in silver fights a redhead in black.",
        )
        self.assertIn("2 clearly defined characters", subset)
        self.assertNotIn("replace", solo.casefold())
        self.assertNotIn("character a", group.casefold())
        self.assertNotIn("character b", group.casefold())

    def test_group_shot_uses_only_its_spatial_primary_reference(self):
        from PIL import Image

        colors = [
            (0, 0, 255), (255, 0, 0), (0, 255, 0),
            (255, 0, 255), (0, 255, 255),
        ]
        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_normalize_recast_person_count",
                "_compose_recast_group_reference_frame",
                "_compose_recast_cast_reference_frame",
                "_recolor_recast_reference_mask",
                "_load_recast_reference_pair",
                "_save_recast_reference_pair",
                "_build_recast_shot_reference_conditioning",
            ),
            {
                "os": os,
                "_RECAST_MASK_COLORS": colors,
            },
        )

        def fake_scene(
            _source, _mask, _count, output_dir, stem, **_kwargs,
        ):
            return {
                "image": os.path.join(output_dir, f"{stem}_scene.png"),
                "mask": os.path.join(output_dir, f"{stem}_scene_mask.png"),
            }

        helpers["_build_recast_source_scene_reference"] = fake_scene
        build = helpers["_build_recast_shot_reference_conditioning"]

        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = []
            mask_paths = []
            expected_colors = []
            primary_items = []
            for mapping_index, color in enumerate(colors[:2]):
                for view_index in range(2):
                    image = np.full(
                        (24, 40, 3),
                        70 + mapping_index * 110 + view_index * 10,
                        dtype=np.uint8,
                    )
                    mask = np.zeros_like(image)
                    mask[3:22, 9:31] = color
                    image_path = os.path.join(
                        temp_dir,
                        f"character_{mapping_index}_{view_index}.png",
                    )
                    mask_path = os.path.join(
                        temp_dir,
                        f"character_{mapping_index}_{view_index}_mask.png",
                    )
                    Image.fromarray(image).save(image_path)
                    Image.fromarray(mask).save(mask_path)
                    image_paths.append(image_path)
                    mask_paths.append(mask_path)
                    expected_colors.append(list(color))
                    if view_index == 0:
                        primary_items.append({
                            "mapping_index": mapping_index,
                            "image": image_path,
                            "mask": mask_path,
                            "color": list(color),
                        })

            prepared = {
                "image_refs": image_paths,
                "primary_mask": mask_paths[0],
                "additional_masks": mask_paths[1:],
                "expected_colors": expected_colors,
                "primary_target_refs": primary_items,
                "reference_canvas": [40, 24],
            }
            source = np.full((24, 40, 3), 45, dtype=np.uint8)
            group_mask = np.zeros_like(source)
            # Deliberately reverse card order in the shot: B is on the left,
            # A is on the right. The single spatial primary must retain that
            # correspondence without a contradictory side-by-side cast sheet.
            group_mask[3:22, 3:15] = colors[1]
            group_mask[3:22, 25:37] = colors[0]

            group = build(
                prepared,
                [0, 1],
                source,
                group_mask,
                temp_dir,
                "job",
                2,
                cooccurring=True,
                reference_canvas=(40, 24),
            )
            solo = build(
                prepared,
                [0],
                source,
                np.where(
                    np.all(group_mask == colors[0], axis=-1)[..., None],
                    np.asarray(colors[0], dtype=np.uint8),
                    np.zeros(3, dtype=np.uint8),
                ),
                temp_dir,
                "job",
                5,
                cooccurring=True,
                reference_canvas=(40, 24),
            )

        self.assertEqual(len(group["image_refs"]), 1)
        self.assertEqual(group["additional_masks"], [])
        self.assertEqual(group["expected_colors"], [None])
        self.assertEqual(group["primary_mode"], "shot_layout")
        self.assertEqual(len(solo["image_refs"]), 3)
        self.assertEqual(len(solo["additional_masks"]), 2)

    def test_shot_manifest_builds_silent_exact_length_internal_tasks(self):
        colors = [
            (0, 0, 255), (255, 0, 0), (0, 255, 0),
            (255, 0, 255), (0, 255, 255),
        ]

        def fake_resize(frames, _canvas, semantic=False):
            return np.asarray(frames, dtype=np.uint8).copy()

        def fake_write(path, _frames, _fps, semantic=False):
            with open(path, "wb") as handle:
                handle.write(b"shot")
            return path

        def fake_references(
            _prepared, active, _frame, _mask, output_dir,
            job_id, shot_index, **_kwargs,
        ):
            stem = os.path.join(
                output_dir,
                f"{job_id}_{shot_index + 1}",
            )
            return {
                "image_refs": [stem + "_ref.png"],
                "primary_mask": stem + "_mask.png",
                "additional_masks": [],
                "expected_colors": [
                    list(colors[0]) if len(active) == 1 else None
                ],
                "clip_identity_ref": stem + "_identity.png",
                "source_scene_reference": {
                    "image": stem + "_scene.png",
                    "mask": stem + "_scene_mask.png",
                },
                "primary_mode": "shot_layout",
            }

        build = _load_functions(
            _LAUNCH_PATH,
            ("_build_recast_shot_manifest",),
            {
                "os": os,
                "_RECAST_MASK_COLORS": colors,
                "_resample_recast_tracking_timeline": self.helpers[
                    "_resample_recast_tracking_timeline"
                ],
                "_plan_recast_shot_segments": self.helpers[
                    "_plan_recast_shot_segments"
                ],
                "_remap_recast_shot_mask": self.helpers[
                    "_remap_recast_shot_mask"
                ],
                "_quantize_recast_shot_frame_count": self.helpers[
                    "_quantize_recast_shot_frame_count"
                ],
                "_build_recast_shot_prompt": self.helpers[
                    "_build_recast_shot_prompt"
                ],
                "_resize_recast_shot_frames": fake_resize,
                "_write_recast_shot_video": fake_write,
                "_build_recast_shot_reference_conditioning": fake_references,
            },
        )["_build_recast_shot_manifest"]

        source = np.full((12, 8, 12, 3), 40, dtype=np.uint8)
        blue = np.zeros_like(source)
        red = np.zeros_like(source)
        blue[0:4, 1:7, 1:5] = colors[0]
        blue[4:8, 1:7, 1:5] = colors[0]
        red[4:8, 1:7, 7:11] = colors[1]
        tracking = {
            "source_frames": source,
            "mapping_masks": [blue, red],
            "shot_ranges": [(0, 4), (4, 8), (8, 12)],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = build(
                {
                    "custom_settings": {},
                    "sliding_window_size": 81,
                    "prompt": (
                        "A blonde woman in silver fights a redhead in black."
                    ),
                },
                tracking,
                {},
                temp_dir,
                "job123",
                reference_canvas=(12, 8),
                target_frame_count=12,
                generation_fps=24,
                minimum_frames=5,
                latent_size=4,
            )

            self.assertEqual(len(result["tasks"]), 2)
            self.assertEqual(len(result["shots"]), 3)
            self.assertEqual(result["shots"][2]["mode"], "passthrough")
            self.assertTrue(
                os.path.isfile(result["shots"][2]["passthrough_path"]),
            )
            for task in result["tasks"]:
                params = task["params"]
                self.assertEqual(params["audio_prompt_type"], "")
                self.assertIsNone(params["audio_source"])
                self.assertEqual(params["video_length"], 5)
                self.assertEqual(params["trim_tail_frames"], 1)
                self.assertEqual(params["force_fps"], "control")
                self.assertNotIn("replace", params["prompt"].casefold())
            self.assertIn(
                "A clearly defined character",
                result["tasks"][0]["params"]["prompt"],
            )
            self.assertEqual(
                result["tasks"][1]["params"]["prompt"],
                "A blonde woman in silver fights a redhead in black.",
            )

    def test_sam3_tracking_segments_are_clamped_and_disjoint(self):
        helpers = _load_functions(
            _SAM3_PREPROCESSOR_PATH,
            (
                "_normalize_sam3_tracking_segments",
                "_sam3_segment_propagation_plan",
                "_is_sam3_tracking_collapse_error",
            ),
        )
        normalize = helpers["_normalize_sam3_tracking_segments"]
        plan = helpers["_sam3_segment_propagation_plan"]
        is_collapse = helpers["_is_sam3_tracking_collapse_error"]

        self.assertEqual(normalize(None, 12), [(0, 12)])
        self.assertEqual(
            normalize([(-3, 5), (4, 9), (9, 30)], 12),
            [(0, 5), (5, 9), (9, 12)],
        )
        with self.assertRaisesRegex(ValueError, "frame pairs"):
            normalize([(0, 4, 8)], 12)

        # Forward max distance is inclusive, while backward excludes the
        # anchor. These plans cover every shot frame without reaching either
        # adjacent camera shot.
        self.assertEqual(plan(0, 85, 0), [("forward", 84)])
        self.assertEqual(
            plan(85, 234, 90),
            [("forward", 143), ("backward", 5)],
        )
        self.assertEqual(plan(550, 586, 585), [("backward", 35)])
        self.assertEqual(plan(10, 11, 10), [])
        with self.assertRaisesRegex(ValueError, "inside"):
            plan(85, 234, 234)

        self.assertTrue(
            is_collapse(RuntimeError("No points are provided")),
        )
        self.assertTrue(
            is_collapse(RuntimeError(
                "The expanded size of the tensor (1) must match the "
                "existing size (0) at non-singleton dimension 1. "
                "Tensor sizes: [5184, 0, 256]"
            )),
        )
        self.assertFalse(is_collapse(RuntimeError("CUDA out of memory")))

    def test_sam3_camera_shots_use_isolated_local_sessions(self):
        import contextlib
        import sys
        from unittest.mock import patch

        if os.path.join(_ROOT, "app") not in sys.path:
            sys.path.insert(0, os.path.join(_ROOT, "app"))
        from preprocessing.sam3 import preprocessor

        class FakePredictor:
            def __init__(self):
                self.session_lengths = {}
                self.started_lengths = []
                self.prompt_frames = []
                self.stream_starts = []
                self.closed_sessions = []
                self.shutdown_called = False

            @staticmethod
            def _outputs(height=4, width=6):
                masks = np.ones((1, height, width), dtype=np.bool_)
                return {
                    "out_binary_masks": masks,
                    "out_obj_ids": np.asarray([0], dtype=np.int64),
                }

            def handle_request(self, request):
                request_type = request["type"]
                if request_type == "start_session":
                    session_id = f"session-{len(self.started_lengths)}"
                    length = len(request["resource_path"])
                    self.started_lengths.append(length)
                    self.session_lengths[session_id] = length
                    return {"session_id": session_id}
                if request_type == "add_prompt":
                    self.prompt_frames.append(request["frame_index"])
                    return {"outputs": self._outputs()}
                if request_type == "close_session":
                    self.closed_sessions.append(request["session_id"])
                    return {"is_success": True}
                raise AssertionError(f"Unexpected request: {request_type}")

            def handle_stream_request(self, request):
                self.stream_starts.append(request["start_frame_index"])
                session_length = self.session_lengths[request["session_id"]]
                start = request["start_frame_index"]
                distance = request["max_frame_num_to_track"]
                if request["propagation_direction"] == "forward":
                    frame_indices = range(
                        start,
                        min(session_length, start + distance + 1),
                    )
                else:
                    frame_indices = range(start - 1, max(-1, start - distance - 1), -1)
                for frame_index in frame_indices:
                    yield {
                        "frame_index": frame_index,
                        "outputs": self._outputs(),
                    }

            def shutdown(self):
                self.shutdown_called = True

        predictor = FakePredictor()
        video = np.zeros((5, 4, 6, 3), dtype=np.uint8)
        with (
            patch.object(preprocessor, "_load_model_builder", return_value=object()),
            patch.object(
                preprocessor,
                "_checkpoint_path",
                return_value=("checkpoint", "sam3.1"),
            ),
            patch.object(preprocessor, "_bpe_path", return_value="bpe"),
            patch.object(preprocessor, "_load_predictor", return_value=predictor),
            patch.object(preprocessor, "_cleanup"),
            patch.object(
                preprocessor,
                "_autocast_context",
                side_effect=lambda: contextlib.nullcontext(),
            ),
            patch.object(preprocessor.torch.cuda, "is_available", return_value=False),
        ):
            mask = preprocessor.run_sam3_video(
                video,
                ["person"],
                preencode_text=False,
                tracking_segments=[(0, 2), (2, 5)],
            )

        self.assertEqual(predictor.started_lengths, [2, 3])
        self.assertEqual(predictor.prompt_frames, [0, 0])
        self.assertEqual(predictor.stream_starts, [0, 0])
        self.assertEqual(
            predictor.closed_sessions,
            ["session-0", "session-1"],
        )
        self.assertTrue(predictor.shutdown_called)
        self.assertTrue(mask.all())

    def test_sam3_grounding_batches_respect_inclusive_shot_bounds(self):
        detector_source = _read(_SAM3_MULTIPLEX_DETECTOR_PATH)
        self.assertGreaterEqual(
            detector_source.count("_sam3_grounding_chunk_bounds("),
            2,
        )
        bounds = _load_functions(
            _SAM3_MULTIPLEX_DETECTOR_PATH,
            ("_sam3_grounding_chunk_bounds",),
        )["_sam3_grounding_chunk_bounds"]

        # Forward distance 84 means frames 0 through 84, inclusive. The
        # former half-open conversion produced an empty batch at frame 84.
        self.assertEqual(
            bounds(
                frame_idx=84,
                num_frames=586,
                batch_size=4,
                max_frame_num_to_track=84,
                propagate_in_video_start_frame_idx=0,
            ),
            (0, 85, 84, 85),
        )
        for frame_index in range(85):
            _, _, chunk_start, chunk_end = bounds(
                frame_idx=frame_index,
                num_frames=586,
                batch_size=4,
                max_frame_num_to_track=84,
                propagate_in_video_start_frame_idx=0,
            )
            self.assertLess(chunk_start, chunk_end)
            self.assertLessEqual(chunk_start, frame_index)
            self.assertLess(frame_index, chunk_end)

        # A later shot starts its batch at frame 85 rather than pulling
        # globally aligned frames 84 and earlier across the camera cut.
        self.assertEqual(
            bounds(
                frame_idx=85,
                num_frames=586,
                batch_size=4,
                max_frame_num_to_track=148,
                propagate_in_video_start_frame_idx=85,
            ),
            (85, 234, 85, 89),
        )
        self.assertEqual(
            bounds(
                frame_idx=233,
                num_frames=586,
                batch_size=4,
                max_frame_num_to_track=148,
                propagate_in_video_start_frame_idx=85,
            ),
            (85, 234, 233, 234),
        )

        # Reverse propagation excludes the prompted anchor and remains inside
        # the same half-open shot range.
        self.assertEqual(
            bounds(
                frame_idx=89,
                num_frames=586,
                batch_size=4,
                max_frame_num_to_track=5,
                propagate_in_video_start_frame_idx=90,
                track_in_reverse=True,
            ),
            (85, 90, 86, 90),
        )
        self.assertEqual(
            bounds(
                frame_idx=85,
                num_frames=586,
                batch_size=4,
                max_frame_num_to_track=5,
                propagate_in_video_start_frame_idx=90,
                track_in_reverse=True,
            ),
            (85, 90, 85, 86),
        )
        with self.assertRaisesRegex(IndexError, "outside propagation"):
            bounds(
                frame_idx=234,
                num_frames=586,
                batch_size=4,
                max_frame_num_to_track=148,
                propagate_in_video_start_frame_idx=85,
            )

    def test_timeline_anchor_supports_separate_character_appearances(self):
        summarize = self.helpers["_summarize_recast_mapping_mask"]
        select = self.helpers["_select_recast_timeline_anchor"]
        blue = np.zeros((10, 12, 20, 3), dtype=np.uint8)
        red = np.zeros_like(blue)
        blue[0:4, 2:10, 2:7] = (0, 0, 255)
        red[6:10, 2:10, 13:18] = (255, 0, 0)

        blue_summary = summarize(blue, fps=10)
        red_summary = summarize(red, fps=10)
        common, scene = select([blue, red])

        self.assertEqual(blue_summary["first_frame_index"], 0)
        self.assertEqual(red_summary["first_frame_index"], 6)
        self.assertAlmostEqual(red_summary["first_time_seconds"], 0.6)
        self.assertIsNone(common)
        self.assertIn(scene, range(10))

    def test_native_group_reference_can_use_a_later_shared_frame(self):
        compose = self.helpers["_compose_recast_native_people_masks"]
        blue = np.asarray((0, 0, 255), dtype=np.uint8)
        red = np.asarray((255, 0, 0), dtype=np.uint8)
        target = np.full((4, 12, 20, 3), 255, dtype=np.uint8)
        target[2:, 2:10, 2:7] = blue
        people = np.zeros_like(target)
        people[2:, 2:10, 2:7] = blue
        people[2:, 2:10, 13:18] = red

        driving, reference, count, assignments = compose(
            target,
            people,
            1,
            reference_frame_index=2,
        )

        self.assertEqual(count, 2)
        self.assertEqual(len(assignments), 1)
        self.assertTrue(np.all(reference[2:10, 2:7] == blue))
        self.assertTrue(np.all(reference[2:10, 13:18] == red))
        self.assertTrue(np.all(driving[0] == 255))

    def test_character_mask_builder_tracks_each_mapping_across_shots(self):
        import sys
        import types
        from unittest.mock import patch

        colors = [(0, 0, 255), (255, 0, 0)]
        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_detect_recast_shot_ranges",
                "_summarize_recast_mapping_mask",
                "_select_recast_timeline_anchor",
                "_normalize_recast_tracking_target",
                "_recast_mask_region",
                "_find_recast_unmapped_shots",
                "_compose_recast_character_masks",
                "_build_recast_character_mask",
            ),
            {
                "_RECAST_MASK_COLORS": colors,
            },
        )
        source = np.empty((12, 24, 40, 3), dtype=np.uint8)
        source[:6] = (10, 30, 50)
        source[6:] = (220, 170, 60)
        calls = []

        def generate(_frames, target, **kwargs):
            calls.append((target, kwargs["tracking_segments"]))
            result = np.zeros_like(source)
            if target == "woman in red":
                result[1:5, 4:20, 3:12] = colors[0]
            elif target == "man in blue":
                result[7:11, 4:20, 28:37] = colors[1]
            return result

        fake_magic_mask = types.SimpleNamespace(
            prepare_video_mask_input=lambda _path: (
                "source.mp4",
                source,
                12.0,
            ),
            generate_keyword_masks=generate,
            save_mask_video=lambda *_args, **_kwargs: "timeline-mask.mp4",
        )
        fake_shared = types.ModuleType("shared")
        fake_shared.magic_mask = fake_magic_mask

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(sys.modules, {"shared": fake_shared}):
                result = helpers["_build_recast_character_mask"](
                    "source.mp4",
                    [
                        {"target": "woman in red"},
                        {"target": "man in blue"},
                    ],
                    temp_dir,
                )

        self.assertEqual(result["video_mask"], "timeline-mask.mp4")
        self.assertEqual(result["shot_ranges"], [[0, 6], [6, 12]])
        self.assertEqual(
            calls,
            [
                ("woman in red", [(0, 6), (6, 12)]),
                ("man in blue", [(0, 6), (6, 12)]),
            ],
        )
        self.assertIsNone(result["common_frame_index"])
        self.assertTrue(result["timeline_aware"])
        self.assertEqual(
            result["mapping_summaries"][1]["first_frame_index"],
            7,
        )

    def test_repaint_masks_can_use_black_driving_background(self):
        compose = self.helpers["_compose_recast_character_masks"]
        mask = np.zeros((2, 8, 12, 3), dtype=np.uint8)
        mask[:, 2:7, 3:8] = (0, 0, 255)

        merged, _ = compose(
            [mask],
            [(0, 0, 255)],
            background_color=(0, 0, 0),
        )

        self.assertTrue(np.all(merged[:, 2:7, 3:8] == (0, 0, 255)))
        self.assertTrue(np.all(merged[:, 0, 0] == 0))

    def test_repaint_regions_and_aspect_are_validated(self):
        normalize = self.helpers["_normalize_repaint_region_mappings"]
        resolution = self.helpers["_repaint_resolution_for_aspect"]
        validate = self.helpers["_validate_repaint_target_aspect"]

        self.assertEqual(normalize(None), [])
        self.assertEqual(
            normalize([{
                "id": "tool",
                "source": "electric wrench",
                "target": "sci-fi gun",
            }]),
            [{
                "id": "tool",
                "source": "electric wrench",
                "target": "sci-fi gun",
            }],
        )
        with self.assertRaisesRegex(ValueError, "needs both"):
            normalize([{"source": "wrench"}])
        self.assertEqual(resolution(1920, 1080), (848, 480))
        self.assertEqual(resolution(1080, 1920), (480, 848))
        self.assertLess(validate(1920, 1080, 1280, 720), 0.001)
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            validate(1920, 1080, 1080, 1920)

    def test_recast_bystander_probe_skips_single_person_full_pass(self):
        import sys
        import types
        from unittest.mock import patch

        colors = [
            (0, 0, 255), (255, 0, 0), (0, 255, 0),
            (255, 0, 255), (0, 255, 255),
        ]
        protection_colors = [*colors, (255, 255, 0)]
        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_normalize_recast_person_count",
                "_recast_color_region",
                "_compose_recast_native_people_masks",
                "_recast_probe_needs_bystander_tracking",
            ),
            {
                "_RECAST_MASK_COLORS": colors,
                "_RECAST_PROTECTION_COLORS": protection_colors,
            },
        )
        target = np.full((10, 14, 3), 255, dtype=np.uint8)
        target[1:9, 2:6] = colors[0]
        people = np.zeros_like(target)
        people[1:9, 2:6] = colors[0]
        fake_magic_mask = types.SimpleNamespace(
            generate_keyword_masks=lambda *_args, **_kwargs: people[None],
        )
        fake_shared = types.ModuleType("shared")
        fake_shared.magic_mask = fake_magic_mask

        with patch.dict(sys.modules, {"shared": fake_shared}):
            self.assertFalse(
                helpers["_recast_probe_needs_bystander_tracking"](
                    np.zeros_like(target),
                    target,
                    1,
                )
            )
            people[1:9, 8:12] = colors[1]
            self.assertTrue(
                helpers["_recast_probe_needs_bystander_tracking"](
                    np.zeros_like(target),
                    target,
                    1,
                )
            )

    def test_recast_prompt_is_detailed_but_not_duplicated(self):
        build_prompt = self.helpers["_build_recast_prompt"]
        prompt = build_prompt("A man in a blue suit dances", 1)
        self.assertTrue(prompt.startswith("A man in a blue suit dances."))
        self.assertIn("The character's face, hair, body shape", prompt)
        self.assertIn("form one coherent, naturally lit scene", prompt)
        self.assertNotIn("Keep ", prompt)
        self.assertEqual(build_prompt(prompt, 1), prompt)
        self.assertIn("performs naturally within the scene", build_prompt("", 1))

        instruction = build_prompt(
            "Replace the woman in red with an obese man in a blue suit", 1,
        )
        self.assertTrue(instruction.startswith("An obese man in a blue suit."))
        self.assertNotIn("Replace the woman", instruction)

        raw_prompt = "replace the woman in red with the man in blue"
        self.assertEqual(build_prompt(raw_prompt, 1, False), raw_prompt)
        with self.assertRaisesRegex(ValueError, "Enter a Recast prompt"):
            build_prompt("   ", 1, False)

    def test_native_people_masks_map_targets_and_bystanders_together(self):
        compose = self.helpers["_compose_recast_native_people_masks"]
        blue = np.asarray((0, 0, 255), dtype=np.uint8)
        red = np.asarray((255, 0, 0), dtype=np.uint8)
        green = np.asarray((0, 255, 0), dtype=np.uint8)
        target_mask = np.full((3, 12, 20, 3), 255, dtype=np.uint8)
        target_mask[:, 2:10, 2:6] = blue
        all_people = np.zeros_like(target_mask)
        all_people[:, 2:10, 2:6] = blue
        all_people[:, 2:10, 8:12] = red
        all_people[:, 2:10, 14:18] = green

        driving, reference, count, assignments = compose(
            target_mask, all_people, 1,
        )

        self.assertEqual(count, 3)
        self.assertEqual(len(assignments), 2)
        self.assertTrue(np.all(driving[:, 2:10, 2:6] == blue))
        self.assertTrue(np.all(driving[:, 2:10, 8:12] == red))
        self.assertTrue(np.all(driving[:, 2:10, 14:18] == green))
        self.assertTrue(np.all(driving[:, 0, 0] == 255))
        self.assertTrue(np.all(reference[2:10, 2:6] == blue))
        self.assertTrue(np.all(reference[2:10, 8:12] == red))
        self.assertTrue(np.all(reference[2:10, 14:18] == green))
        self.assertTrue(np.all(reference[0, 0] == 0))

    def test_group_reference_replaces_old_target_and_retains_bystander(self):
        compose = self.helpers["_compose_recast_group_reference_frame"]
        blue = np.asarray((0, 0, 255), dtype=np.uint8)
        red = np.asarray((255, 0, 0), dtype=np.uint8)

        source = np.full((12, 20, 3), (30, 40, 50), dtype=np.uint8)
        source[2:10, 2:6] = (220, 10, 10)  # old selected subject
        source[2:10, 14:18] = (10, 180, 40)  # original bystander
        semantic = np.zeros_like(source)
        semantic[2:10, 2:6] = blue
        semantic[2:10, 14:18] = red

        replacement = np.full_like(source, 127)
        replacement[2:10, 8:12] = (190, 190, 215)
        replacement_mask = np.zeros_like(source)
        replacement_mask[2:10, 8:12] = blue

        composite, composite_mask = compose(
            source,
            semantic,
            [(replacement, replacement_mask)],
            1,
        )

        self.assertTrue(np.all(composite[5, 3] == (190, 190, 215)))
        self.assertTrue(np.all(composite[5, 15] == (10, 180, 40)))
        self.assertTrue(np.all(composite[0, 0] == 127))
        self.assertFalse(
            bool(np.all(composite == (220, 10, 10), axis=-1).any()),
        )
        self.assertTrue(np.all(composite_mask[5, 3] == blue))
        self.assertTrue(np.all(composite_mask[5, 15] == red))
        self.assertTrue(np.all(composite_mask[0, 0] == 0))

    def test_source_scene_conditioning_uses_a_stable_hidden_reference_pair(self):
        from PIL import Image

        enable_scene = self.helpers[
            "_enable_recast_dynamic_source_scene_reference"
        ]
        blue = [0, 0, 255]
        source = np.full((40, 60, 3), (18, 44, 73), dtype=np.uint8)
        source[10:34, 20:42] = (225, 15, 20)
        target_mask = np.zeros_like(source)
        target_mask[10:34, 20:42] = blue

        with tempfile.TemporaryDirectory() as temp_dir:
            replacement_path = os.path.join(temp_dir, "replacement.png")
            face_path = os.path.join(temp_dir, "face-detail.png")
            face_mask_path = os.path.join(
                temp_dir, "face-detail-mask.png",
            )
            Image.new("RGB", (60, 40), (80, 90, 100)).save(
                replacement_path,
            )
            Image.new("RGB", (60, 40), (110, 120, 130)).save(face_path)
            Image.new("RGB", (60, 40), tuple(blue)).save(face_mask_path)
            params = {
                "resolution": "60x40",
                "edit_recast_reference_canvas": [896, 512],
                "image_refs": [replacement_path, face_path],
                "custom_settings": {
                    "scail2_additional_reference_mask_paths": [
                        face_mask_path,
                    ],
                    "scail2_reference_expected_colors": [blue, blue],
                },
            }
            scene = enable_scene(
                params,
                source,
                target_mask,
                1,
                temp_dir,
                "grey-regression",
            )
            with Image.open(scene["image"]) as image:
                scene_rgb = np.asarray(image.convert("RGB"))
            with Image.open(scene["mask"]) as image:
                scene_mask = np.asarray(image.convert("RGB"))

        center_y = scene_rgb.shape[0] // 2
        center_x = scene_rgb.shape[1] // 2
        self.assertEqual(scene_rgb.shape[:2], (512, 896))
        self.assertTrue(np.all(scene_rgb[0, 0] == (18, 44, 73)))
        self.assertLessEqual(
            int(np.max(np.abs(
                scene_rgb[center_y, center_x].astype(np.int16)
                - np.asarray((18, 44, 73), dtype=np.int16)
            ))),
            16,
        )
        self.assertFalse(
            np.all(scene_rgb[center_y, center_x] == (225, 15, 20)),
        )
        self.assertTrue(np.all(scene_mask[0, 0] == 255))
        self.assertTrue(np.all(scene_mask[center_y, center_x] == 0))
        self.assertEqual(
            params["image_refs"],
            [replacement_path, face_path],
        )
        self.assertEqual(
            params["custom_settings"][
                "scail2_additional_reference_mask_paths"
            ],
            [face_mask_path],
        )
        self.assertEqual(
            params["custom_settings"]["scail2_reference_expected_colors"],
            [blue, blue],
        )
        self.assertIs(
            params["custom_settings"][
                "scail2_dynamic_source_scene_reference"
            ],
            True,
        )
        self.assertEqual(
            params["custom_settings"][
                "scail2_source_scene_reference_path"
            ],
            scene["image"],
        )
        self.assertEqual(
            params["custom_settings"]["scail2_source_scene_mask_path"],
            scene["mask"],
        )
        self.assertEqual(
            params["edit_recast_source_scene_reference"],
            scene["image"],
        )
        self.assertEqual(
            params["edit_recast_source_scene_conditioning"],
            "stable_official_reference",
        )
        self.assertEqual(
            params["edit_recast_reference_canvas"],
            [896, 512],
        )

    def test_native_bystander_reference_removes_target_and_scene_rgb(self):
        matte = self.helpers["_matte_recast_reference_frame"]
        frame = np.zeros((8, 12, 3), dtype=np.uint8)
        frame[:] = (10, 20, 30)
        frame[1:7, 1:5] = (200, 10, 10)  # old selected subject
        frame[1:7, 7:11] = (10, 200, 10)  # preserved bystander
        semantic_mask = np.zeros_like(frame)
        semantic_mask[1:7, 7:11] = (255, 0, 0)

        matted = matte(frame, semantic_mask)

        self.assertTrue(np.all(matted[1:7, 7:11] == (10, 200, 10)))
        self.assertTrue(np.all(matted[1:7, 1:5] == 127))
        self.assertTrue(np.all(matted[0, 0] == 127))

    def test_native_group_reference_resolves_first_frame_overlap_to_target(self):
        compose = self.helpers["_compose_recast_native_people_masks"]
        blue = np.asarray((0, 0, 255), dtype=np.uint8)
        red = np.asarray((255, 0, 0), dtype=np.uint8)
        target_mask = np.full((2, 10, 14, 3), 255, dtype=np.uint8)
        target_mask[:, 1:9, 2:6] = blue
        all_people = np.zeros_like(target_mask)
        all_people[:, 1:9, 2:6] = blue
        # A real bystander partly crosses behind the selected subject.
        all_people[:, 1:9, 4:10] = red

        driving, reference, count, _ = compose(target_mask, all_people, 1)

        self.assertEqual(count, 2)
        self.assertTrue(np.all(driving[:, 1:9, 2:6] == blue))
        self.assertTrue(np.all(reference[1:9, 2:6] == blue))
        self.assertTrue(np.all(reference[1:9, 6:10] == red))

    def test_resolved_seed_is_read_from_output_filename(self):
        extract_seed = self.helpers["_extract_output_seed"]
        self.assertEqual(
            extract_seed("2026-07-21-17h39m34s_seed690270001_prompt.mp4"),
            690270001,
        )
        self.assertIsNone(extract_seed("rejoin_multiclip.mp4"))

    def test_identity_latent_blend_can_follow_reordered_group_reference(self):
        resolve_index = _load_functions(
            _SCAIL2_PATH,
            ("_get_scail2_identity_latent_reference_index",),
        )["_get_scail2_identity_latent_reference_index"]

        self.assertEqual(resolve_index(None), 0)
        self.assertEqual(resolve_index({}), 0)
        self.assertEqual(
            resolve_index({"scail2_identity_latent_reference_index": 1}),
            1,
        )
        self.assertEqual(
            resolve_index({"scail2_identity_latent_reference_index": "bad"}),
            0,
        )

    def test_aligned_reference_uses_the_overlapping_new_person_silhouette(self):
        align = self.helpers["_align_recast_reference_mask"]
        blue = (0, 0, 255)
        red = (255, 0, 0)
        source = np.zeros((8, 12, 3), dtype=np.uint8)
        source[2:7, 1:4] = blue
        reference = np.zeros_like(source)
        # Replacement is wider than the source target but overlaps it.
        reference[1:8, 0:6] = red
        # A bystander is larger but lives elsewhere in the frame.
        reference[0:8, 8:12] = blue

        aligned, matched = align(source, reference, [blue], [blue, red])

        self.assertEqual(matched, 1)
        self.assertTrue(np.all(aligned[1:8, 0:6] == np.asarray(blue)))
        self.assertFalse(bool(aligned[:, 8:12].any()))

    def test_aligned_reference_preparation_runs_with_the_exact_mask(self):
        import sys
        import tempfile
        import types
        from unittest.mock import patch
        from PIL import Image

        colors = [
            (0, 0, 255), (255, 0, 0), (0, 255, 0),
            (255, 0, 255), (0, 255, 255),
        ]
        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_align_recast_reference_mask",
                "_recast_subject_crop_box",
                "_fit_recast_reference_layers",
                "_prepare_recast_reference_frame",
                "_recast_image_data_uri",
                "_recast_reference_preview_payload",
                "_recast_should_add_auto_face_detail",
                "_prepare_recast_reference_conditioning",
            ),
            {
                "os": os,
                "_RECAST_MASK_COLORS": colors,
            },
        )
        source_mask = np.zeros((48, 80, 3), dtype=np.uint8)
        source_mask[8:44, 10:35] = colors[0]
        reference_mask = np.zeros_like(source_mask)
        reference_mask[5:46, 7:40] = colors[1]

        fake_magic_mask = types.SimpleNamespace(
            generate_keyword_masks=lambda *_args, **_kwargs: reference_mask[None],
        )
        fake_shared = types.ModuleType("shared")
        fake_shared.magic_mask = fake_magic_mask

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = os.path.join(temp_dir, "edited-first-frame.png")
            Image.new("RGB", (80, 48), (40, 80, 120)).save(reference_path)
            with patch.dict(sys.modules, {"shared": fake_shared}):
                prepared = helpers["_prepare_recast_reference_conditioning"](
                    [{
                        "ref_image_path": reference_path,
                        "additional_ref_image_paths": [],
                        "reference_aligned_to_source": True,
                    }],
                    source_mask,
                    temp_dir,
                    "aligned-test",
                    isolate_reference=False,
                    selected_count=1,
                )

            self.assertTrue(os.path.isfile(prepared["image_refs"][0]))
            self.assertTrue(os.path.isfile(prepared["clip_identity_ref"]))
            self.assertTrue(os.path.isfile(prepared["primary_mask"]))
            self.assertEqual(prepared["expected_colors"], [[0, 0, 255]])
            with Image.open(prepared["primary_mask"]) as mask_image:
                mask = np.asarray(mask_image.convert("RGB"))
            self.assertTrue(bool(np.all(mask == colors[0], axis=-1).any()))

    def test_auto_face_detail_frames_the_upper_identity_region(self):
        crop_box = _load_functions(
            _LAUNCH_PATH,
            ("_recast_face_detail_crop_box",),
        )["_recast_face_detail_crop_box"]
        subject = np.zeros((2000, 1332), dtype=bool)
        subject[20:1980, 40:1290] = True

        left, top, right, bottom = crop_box(subject)

        self.assertEqual(top, 0)
        self.assertGreaterEqual(left, 200)
        self.assertLessEqual(right, 1130)
        self.assertGreaterEqual(right - left, 850)
        self.assertLessEqual(right - left, 900)
        self.assertGreaterEqual(bottom - top, 1070)
        self.assertLessEqual(bottom - top, 1120)

    def test_auto_face_detail_persists_a_transparent_derived_view(self):
        import tempfile
        from PIL import Image

        subject = np.zeros((200, 132), dtype=bool)
        subject[2:198, 4:128] = True

        def fake_subject_mask(_rgb, _alpha):
            return subject, subject.astype(np.float32), "test subject mask"

        def fake_refine(rgb, region, opacity, source, **_kwargs):
            return region, opacity, source, rgb

        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_recast_face_detail_crop_box",
                "_derive_recast_face_detail_reference",
            ),
            {
                "_recast_reference_subject_mask": fake_subject_mask,
                "_refine_recast_reference_cutout": fake_refine,
            },
        )
        rgba = np.zeros((200, 132, 4), dtype=np.uint8)
        rgba[..., :3] = (20, 80, 160)
        rgba[..., 3] = 255
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "primary.png")
            detail_path = os.path.join(temp_dir, "detail.png")
            Image.fromarray(rgba).save(source_path)

            result = helpers["_derive_recast_face_detail_reference"](
                source_path, detail_path,
            )

            self.assertIsNotNone(result)
            metadata = result["metadata"]
            self.assertTrue(os.path.isfile(detail_path))
            self.assertEqual(
                metadata["detail_source"],
                "upper-subject crop from test subject mask",
            )
            self.assertNotEqual(metadata["crop_box"], [0, 0, 132, 200])
            with Image.open(detail_path) as detail:
                self.assertEqual(detail.mode, "RGBA")
                self.assertEqual(list(detail.size), metadata["detail_size"])
            self.assertEqual(
                result["detail_conditioning"][0].shape,
                (metadata["detail_size"][1], metadata["detail_size"][0]),
            )

    def test_auto_face_detail_reuses_one_mask_for_both_conditioning_views(self):
        import sys
        import tempfile
        import types
        from unittest.mock import patch
        from PIL import Image

        colors = [
            (0, 0, 255), (255, 0, 0), (0, 255, 0),
            (255, 0, 255), (0, 255, 255),
        ]
        calls = {"count": 0}

        def fake_subject_mask(rgb, _alpha):
            calls["count"] += 1
            height, width = rgb.shape[:2]
            subject = np.zeros((height, width), dtype=bool)
            subject[2:height - 2, 4:width - 4] = True
            return subject, subject.astype(np.float32), "test subject mask"

        def fake_refine(rgb, region, opacity, source, **_kwargs):
            return region, opacity, source, rgb

        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_recast_subject_crop_box",
                "_recast_face_detail_crop_box",
                "_derive_recast_face_detail_reference",
                "_recast_region_bbox_aspect",
                "_recast_face_detail_upscale_factor",
                "_recast_should_add_auto_face_detail",
                "_fit_recast_reference_layers",
                "_prepare_recast_reference_frame",
                "_recast_image_data_uri",
                "_recast_reference_preview_payload",
                "_prepare_recast_reference_conditioning",
            ),
            {
                "os": os,
                "_RECAST_MASK_COLORS": colors,
                "_recast_reference_subject_mask": fake_subject_mask,
                "_refine_recast_reference_cutout": fake_refine,
            },
        )
        fake_shared = types.ModuleType("shared")
        fake_shared.magic_mask = types.SimpleNamespace()

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = os.path.join(temp_dir, "primary.png")
            Image.new("RGB", (660, 1000), (40, 80, 120)).save(reference_path)
            with patch.dict(sys.modules, {"shared": fake_shared}):
                prepared = helpers["_prepare_recast_reference_conditioning"](
                    [{
                        "ref_image_path": reference_path,
                        "additional_ref_image_paths": [],
                        "reference_aligned_to_source": False,
                    }],
                    np.zeros((48, 80, 3), dtype=np.uint8),
                    temp_dir,
                    "auto-detail-test",
                    isolate_reference=True,
                    selected_count=1,
                    auto_face_detail=True,
                )

            self.assertEqual(calls["count"], 1)
            self.assertEqual(len(prepared["image_refs"]), 2)
            self.assertEqual(len(prepared["additional_masks"]), 1)
            self.assertEqual(
                prepared["expected_colors"],
                [[0, 0, 255], [0, 0, 255]],
            )
            self.assertEqual(len(prepared["auto_face_detail_refs"]), 1)
            self.assertEqual(
                [preview["kind"] for preview in prepared["previews"]],
                ["primary", "auto_face_detail"],
            )

    def test_auto_face_detail_defers_to_explicit_or_aligned_views(self):
        should_add = _load_functions(
            _LAUNCH_PATH,
            (
                "_recast_face_detail_upscale_factor",
                "_recast_should_add_auto_face_detail",
            ),
        )["_recast_should_add_auto_face_detail"]

        self.assertTrue(should_add({
            "reference_aligned_to_source": False,
            "additional_ref_image_paths": [],
        }))
        self.assertFalse(should_add({
            "reference_aligned_to_source": True,
            "additional_ref_image_paths": [],
        }))
        self.assertFalse(should_add({
            "reference_aligned_to_source": False,
            "additional_ref_image_paths": ["manual-close-up.png"],
        }))
        self.assertFalse(should_add({
            "reference_aligned_to_source": False,
            "additional_ref_image_paths": [],
        }, enabled=False))

    def test_auto_face_detail_skips_competing_full_body_spatial_view(self):
        colors = [(0, 0, 255)]
        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_recast_region_bbox_aspect",
                "_recast_face_detail_upscale_factor",
                "_recast_should_add_auto_face_detail",
            ),
        )
        mapping = {
            "reference_aligned_to_source": False,
            "additional_ref_image_paths": [],
        }
        full_body_target = np.zeros((100, 100, 3), dtype=np.uint8)
        full_body_target[5:95, 40:60] = colors[0]
        upper_body_reference = np.zeros((100, 100), dtype=bool)
        upper_body_reference[5:95, 25:75] = True

        should_add = helpers["_recast_should_add_auto_face_detail"]
        self.assertFalse(should_add(
            mapping,
            source_probe_mask=full_body_target,
            semantic_color=colors[0],
            reference_subject=upper_body_reference,
        ))

        talking_head_target = np.zeros((100, 100, 3), dtype=np.uint8)
        talking_head_target[15:85, 20:80] = colors[0]
        talking_head_reference = np.zeros((100, 100), dtype=bool)
        talking_head_reference[15:85, 20:80] = True
        self.assertTrue(should_add(
            mapping,
            source_probe_mask=talking_head_target,
            semantic_color=colors[0],
            reference_subject=talking_head_reference,
        ))

    def test_auto_face_detail_skips_excessive_crop_enlargement(self):
        helpers = _load_functions(
            _LAUNCH_PATH,
            (
                "_recast_face_detail_upscale_factor",
                "_recast_should_add_auto_face_detail",
            ),
        )
        mapping = {
            "reference_aligned_to_source": False,
            "additional_ref_image_paths": [],
        }
        upscale = helpers["_recast_face_detail_upscale_factor"]
        should_add = helpers["_recast_should_add_auto_face_detail"]

        self.assertAlmostEqual(
            upscale((208, 260), (480, 640)),
            480 / 208,
            places=5,
        )
        self.assertFalse(should_add(
            mapping,
            detail_size=(208, 260),
            canvas_size=(480, 640),
        ))
        self.assertTrue(should_add(
            mapping,
            detail_size=(506, 633),
            canvas_size=(480, 640),
        ))

    def test_recast_wires_count_into_process_and_sam3(self):
        source = _read(_LAUNCH_PATH)
        self.assertIn('"video_prompt_type": f"V0{person_count}AI"', source)
        self.assertIn('"image_prompt_type": ""', source)
        self.assertNotIn('"image_start": ref_image_path', source)
        self.assertIn('"num_inference_steps": inference_steps', source)
        self.assertIn(
            'body.get("num_inference_steps"),\n        8 if recast_fast else 40',
            source,
        )
        self.assertIn('"flow_shift": 1 if recast_fast', source)
        self.assertIn('"edit_recast_person_count": person_count', source)
        self.assertIn("color_palette=mask_colors", source)
        self.assertIn("max_colored_objects=person_count", source)
        self.assertIn('background_color=(255, 255, 255)', source)

    def test_strict_pixel_lock_is_an_opt_in_fallback(self):
        launch = _read(_LAUNCH_PATH)
        store = _read(_STORE_PATH)
        controls = _read(_RECAST_CONTROLS_PATH)
        self.assertIn('body.get("protect_bystanders") is True', launch)
        self.assertIn('"_recast_protect_bystanders": protect_bystanders', launch)
        self.assertIn('colorkey=0xFFFFFF', launch)
        self.assertNotIn('"_recast_mask_background"', launch)
        self.assertIn("_build_recast_adaptive_mask(", launch)
        self.assertIn('wgp.release_model()', launch)
        self.assertIn("source_video, adaptive_mask, generated_video", launch)
        self.assertIn("Protecting people outside the generated replacement", launch)
        self.assertIn("editRecastProtectBystanders: boolean", store)
        self.assertIn("editRecastProtectBystanders: false", store)
        self.assertIn("protect_bystanders: false", store)
        self.assertNotIn("Strict pixel lock outside replacement", controls)

    def test_recast_offers_single_reference_native_group_conditioning(self):
        launch = _read(_LAUNCH_PATH)
        scail2 = _read(_SCAIL2_PATH)
        wan_handler = _read(_WAN_HANDLER_PATH)
        store = _read(_STORE_PATH)
        controls = _read(_RECAST_CONTROLS_PATH)
        self.assertIn('if "preserve_bystanders" in body:', launch)
        self.assertIn('body.get("preserve_bystanders") is True', launch)
        self.assertIn('body.get("preserve_scene_reference") is True', launch)
        self.assertIn('preserve_bystanders = True', launch)
        self.assertIn("_build_recast_native_people_conditioning(", launch)
        self.assertIn("_compose_recast_group_reference_frame(", launch)
        self.assertIn("recast_group_ref_", launch)
        self.assertIn("recast_group_mask_", launch)
        self.assertIn('"scail2_primary_reference_people": 1', launch)
        self.assertIn('"scail2_additional_reference_mask_paths"', launch)
        self.assertIn('job["params"]["image_refs"] = [', launch)
        self.assertIn("[None, *target_expected_colors]", launch)
        self.assertIn('"scail2_identity_latent_reference_index"', launch)
        self.assertIn("_get_scail2_identity_latent_reference_index(", scail2)
        self.assertIn(
            '"scail2_identity_latent_reference_index"',
            wan_handler,
        )
        self.assertIn("native_people['conditioning_count']", launch)
        self.assertNotIn("recast_bystanders_matted_ref_", launch)
        self.assertIn("_build_recast_source_scene_reference(", launch)
        self.assertIn(
            "_enable_recast_dynamic_source_scene_reference(",
            launch,
        )
        self.assertIn("recast_source_scene_ref_", launch)
        self.assertIn("recast_source_scene_mask_", launch)
        self.assertIn("_ensure_source_scene_reference(", launch)
        self.assertIn(
            '"edit_recast_source_scene_reference"',
            launch,
        )
        self.assertIn(
            '"scail2_dynamic_source_scene_reference"',
            launch,
        )
        self.assertIn(
            '"scail2_dynamic_source_scene_reference"',
            wan_handler,
        )
        self.assertIn(
            '"scail2_primary_only_continuations"',
            wan_handler,
        )
        self.assertIn(
            "def _build_dynamic_source_scene_reference",
            scail2,
        )
        self.assertIn("def _load_additional_ref_mask", scail2)
        self.assertIn('"scail2_primary_reference_people"', scail2)
        self.assertIn("binary_background = black | white", scail2)
        self.assertIn("editRecastPreserveBystanders: true", store)
        self.assertIn("preserve_bystanders: true", store)
        self.assertIn("p.edit_recast_preserve_scene_reference === true", store)
        self.assertIn("_recast_probe_needs_bystander_tracking(", launch)
        self.assertIn("if map_native_bystanders:", launch)
        self.assertIn("skipping the second full SAM3 pass", launch)
        self.assertNotIn("Preserve other people natively", controls)
        self.assertIn("preserves detected bystanders automatically", controls)

    def test_recast_discovers_mappings_across_the_selected_timeline(self):
        launch = _read(_LAUNCH_PATH)
        scail2 = _read(_SCAIL2_PATH)
        wan_handler = _read(_WAN_HANDLER_PATH)
        controls = _read(_RECAST_CONTROLS_PATH)
        client = _read(_API_CLIENT_PATH)
        magic_mask = _read(_MAGIC_MASK_PATH)
        sam3 = _read(_SAM3_PREPROCESSOR_PATH)

        self.assertIn("def _detect_recast_shot_ranges(", launch)
        self.assertIn("tracking_segments=shot_ranges", launch)
        self.assertIn("def _recover_recast_unmapped_shots(", launch)
        self.assertIn(
            "running generic-person appearance recovery",
            launch,
        )
        self.assertIn("anywhere in the selected video", launch)
        self.assertIn('"edit_recast_timeline_anchors"', launch)
        self.assertIn('"scail2_timeline_source_scene_reference"', launch)
        self.assertIn('"scail2_timeline_source_scene_reference"', scail2)
        self.assertIn(
            '"scail2_timeline_source_scene_reference"',
            wan_handler,
        )
        self.assertIn("and not timeline_scene_reference", scail2)
        self.assertIn("end_time: editEndTime", controls)
        self.assertIn("Not found in the selected timeline.", controls)
        self.assertIn("end_time?: number;", client)
        self.assertIn("tracking_segments=tracking_segments", magic_mask)
        self.assertIn("timeline_segments =", sam3)
        self.assertIn("for shot_start, shot_end in timeline_segments", sam3)
        self.assertIn("object_color_map.clear()", sam3)
        self.assertIn(
            "propagation_plan = _sam3_segment_propagation_plan(",
            sam3,
        )
        self.assertIn(
            "def _build_recast_target_group_reference(",
            launch,
        )
        self.assertIn(
            '"scail2_reference_expected_colors"\n                    ] = [\n                        None,',
            launch,
        )
        self.assertIn(
            "_select_scail2_window_reference_indices(",
            scail2,
        )
        self.assertIn(
            "def _build_recast_shot_manifest(",
            launch,
        )
        self.assertIn(
            "def _run_recast_shot_generation(",
            launch,
        )
        self.assertIn(
            '"_recast_shot_manifest"',
            launch,
        )
        self.assertIn(
            '"defer_concat": True',
            launch,
        )
        self.assertIn(
            'and not multi_clip_info.get("defer_concat", False)',
            _read(_WGP_PATH),
        )
        self.assertIn(
            "Routed this window's character references",
            scail2,
        )
        self.assertNotIn(
            '"max_frame_num_to_track": shot_end - shot_start',
            sam3,
        )

    def test_recast_worker_initializes_probe_frame_before_local_use(self):
        tree = ast.parse(_read(_LAUNCH_PATH), filename="app/launch.py")
        endpoint = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "recast_endpoint"
        )
        worker = next(
            node
            for node in endpoint.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_recast"
        )
        stores = [
            node.lineno
            for node in ast.walk(worker)
            if isinstance(node, ast.Name)
            and node.id == "probe_frame"
            and isinstance(node.ctx, ast.Store)
        ]
        loads = [
            node.lineno
            for node in ast.walk(worker)
            if isinstance(node, ast.Name)
            and node.id == "probe_frame"
            and isinstance(node.ctx, ast.Load)
        ]

        self.assertTrue(stores)
        self.assertTrue(loads)
        self.assertLess(min(stores), min(loads))

    def test_recast_relighting_is_hash_pinned_and_opt_in(self):
        launch = _read(_LAUNCH_PATH)
        store = _read(_STORE_PATH)
        controls = _read(_RECAST_CONTROLS_PATH)
        selector = _read(_LORA_SELECTOR_PATH)
        self.assertIn(
            '_RECAST_RELIGHTING_LORA_FILENAME = "scail2_relighting_lora.safetensors"',
            launch,
        )
        self.assertIn(
            '"80d338a7969c1b286c8f5c4996b37eb198d0864837fecb6c87c106ca74571a2b"',
            launch,
        )
        self.assertIn('"converter": "scail2_sat_lora"', launch)
        self.assertIn('body.get("use_relighting") is True', launch)
        self.assertIn("def _normalize_recast_lora_settings(", launch)
        self.assertIn("single-phase LoRA settings", launch)
        self.assertIn("editRecastUseRelighting: false", store)
        self.assertIn("use_relighting: state.editRecastUseRelighting", store)
        self.assertIn("const recastSinglePhase =", store)
        self.assertIn("const recastSinglePhase =", selector)
        self.assertIn("const phases = recastSinglePhase ? 1", selector)
        self.assertIn("Match lighting", controls)
        self.assertIn("official SCAIL-2 Relighting LoRA", controls)
        self.assertIn("single strength control", controls)

    def test_recast_isolates_reference_background_by_default(self):
        launch = _read(_LAUNCH_PATH)
        scail2 = _read(_SCAIL2_PATH)
        store = _read(_STORE_PATH)
        controls = _read(_RECAST_CONTROLS_PATH)
        self.assertIn('body.get("isolate_reference") is not False', launch)
        self.assertIn('"scail2_isolate_reference_background": False', launch)
        self.assertIn('"scail2_reference_alpha_path"', launch)
        self.assertIn('"edit_recast_isolate_reference": isolate_reference', launch)
        self.assertIn("def _prepare_recast_reference_frame", launch)
        self.assertIn("isolate_reference=isolate_reference", launch)
        self.assertIn("def _load_reference_alpha", scail2)
        self.assertIn("def _isolate_scail2_reference", scail2)
        self.assertIn("image_ref = _isolate_scail2_reference(", scail2)
        self.assertIn("editRecastIsolateReference: true", store)
        self.assertIn("isolate_reference: true", store)
        self.assertNotIn("Isolate replacement from reference background", controls)
        self.assertIn("Maestro isolates references", controls)

    def test_recast_uses_identity_latents_and_trims_motion_preroll(self):
        launch = _read(_LAUNCH_PATH)
        scail2 = _read(_SCAIL2_PATH)
        any2video = _read(os.path.join(
            _ROOT, "app", "models", "wan", "any2video.py",
        ))
        wgp = _read(_WGP_PATH)
        client = _read(os.path.join(_ROOT, "ui", "src", "api", "client.ts"))

        self.assertIn('"identity_image": identity_image', launch)
        self.assertIn('"scail2_clip_reference_path"', launch)
        self.assertIn(
            '"scail2_identity_latent_isolation": isolate_reference',
            launch,
        )
        self.assertIn('"scail2_recast_warmup_frames": _RECAST_WARMUP_FRAMES', launch)
        self.assertIn('"scail2_primary_only_continuations": True', launch)
        self.assertIn('"video_length": gen_frames', launch)
        self.assertIn("def _load_clip_identity_reference", scail2)
        self.assertIn("def _blend_scail2_identity_latents", scail2)
        self.assertIn(
            'custom_settings.get("scail2_identity_latent_isolation") is True',
            scail2,
        )
        self.assertIn(
            "def _use_scail2_primary_only_continuation_refs",
            scail2,
        )
        self.assertIn('"clip_image_start": clip_identity_ref.squeeze(1)', scail2)
        self.assertIn('"post_decode_pre_trim"', scail2)
        self.assertIn(
            'scail2_conditioning.get("post_decode_pre_trim", 0)',
            any2video,
        )
        self.assertIn("def _shift_guide_window_for_warmup", wgp)
        self.assertIn("def _prepend_reverse_motion_preroll", wgp)
        self.assertIn("inputs[\"video_length\"] = published_video_length", wgp)
        self.assertIn("clip_identity_image?: string", client)

    def test_adaptive_protection_keeps_new_subject_shape_but_rejects_spill(self):
        select = self.helpers["_select_recast_generated_regions"]
        blue = np.asarray((0, 0, 255), dtype=np.uint8)
        red = np.asarray((255, 0, 0), dtype=np.uint8)
        source = np.full((2, 8, 14, 3), 255, dtype=np.uint8)
        source[:, 2:7, 1:4] = blue

        generated = np.zeros_like(source)
        # The intended replacement is wider than the source target.
        generated[:, 1:8, 0:7] = blue
        # A separate bystander was also changed by the model.
        generated[:, 1:8, 10:14] = red

        adaptive, scores = select(source, generated, 1)

        self.assertEqual(scores[0][0], 0)
        self.assertTrue(bool(adaptive[:, 1:8, 0:7].all()))
        self.assertFalse(bool(adaptive[:, :, 10:14].any()))

    def test_recast_default_prefers_dedicated_fast_scail2(self):
        launch = _read(_LAUNCH_PATH)
        store = _read(_STORE_PATH)
        controls = _read(_RECAST_CONTROLS_PATH)
        self.assertIn('body.get("model_type") or _RECAST_FAST_MODEL_TYPE', launch)
        self.assertIn("? 'scail2_14B_recast_fast'", store)
        self.assertIn("Fast is recommended (8 steps)", controls)
        self.assertIn("HQ uses the full 40-step schedule", controls)
        self.assertIn("initialModelType === 'scail2_14B_fast'", store)
        self.assertIn("editSubMode: 'restyle' as const", store)
        self.assertIn("editSubMode: 'recast' as const", store)

    def test_recast_resolution_profile_does_not_change_model_schedule(self):
        launch = _read(_LAUNCH_PATH)
        store = _read(_STORE_PATH)
        controls = _read(_RECAST_CONTROLS_PATH)
        selector = _read(_SCAIL_RESOLUTION_SELECTOR_PATH)
        client = _read(os.path.join(_ROOT, "ui", "src", "api", "client.ts"))

        self.assertIn('"512p": (896, 512)', launch)
        self.assertIn('"704p": (1280, 704)', launch)
        self.assertIn('"resolution": recast_resolution', launch)
        self.assertIn('"sliding_window_size": recast_window_size', launch)
        self.assertIn(
            'body.get("num_inference_steps")',
            launch,
        )
        self.assertIn('"num_inference_steps": inference_steps', launch)
        self.assertNotIn('"num_inference_steps": 8 if recast_fast else', launch)
        self.assertIn(
            '"edit_recast_resolution_profile": resolution_profile',
            launch,
        )
        self.assertNotIn('if resolution_profile == "512p"', launch)
        self.assertIn("editRecastResolutionProfile: '480p'", store)
        self.assertIn(
            "resolution_profile: state.editRecastResolutionProfile",
            store,
        )
        self.assertIn(
            "p.edit_recast_resolution_profile === '512p'",
            store,
        )
        self.assertIn(
            "p.edit_recast_resolution_profile === '704p'",
            store,
        )
        self.assertIn("<ScailResolutionSelector", controls)
        self.assertIn("detail: 'Balanced'", selector)
        self.assertIn("detail: 'Experimental'", selector)
        self.assertIn("896×512", selector)
        self.assertIn("up to 1280×704", selector)
        self.assertIn("33–49 frame windows", selector)
        self.assertIn("totalVramGb < 16", selector)
        self.assertIn("inference steps, or guidance", selector)
        self.assertIn(
            "resolution_profile?: ScailResolutionProfile",
            client,
        )
        self.assertIn("sliding_window_size?: number", client)

    def test_repaint_shares_resolution_profiles_and_adaptive_windows(self):
        launch = _read(_LAUNCH_PATH)
        store = _read(_STORE_PATH)
        controls = _read(_REPAINT_CONTROLS_PATH)

        self.assertIn(
            "output_width, output_height = _recast_resolution_for_source(",
            launch,
        )
        self.assertIn(
            "repaint_window_size = _recast_window_size_for_profile(",
            launch,
        )
        self.assertIn('"sliding_window_size": repaint_window_size', launch)
        self.assertIn(
            '"edit_repaint_resolution_profile": resolution_profile',
            launch,
        )
        self.assertIn("editRepaintResolutionProfile: '480p'", store)
        self.assertIn(
            "resolution_profile: state.editRepaintResolutionProfile",
            store,
        )
        self.assertIn(
            "p.edit_repaint_resolution_profile === '704p'",
            store,
        )
        self.assertIn("<ScailResolutionSelector", controls)
        self.assertIn('workflow="Repaint"', controls)

    def test_scail_edit_advanced_panel_only_shows_effective_settings(self):
        advanced = _read(_ADVANCED_SETTINGS_PATH)
        launch = _read(_LAUNCH_PATH)
        store = _read(_STORE_PATH)

        self.assertIn("const isScailEdit = isRecast || isRepaint", advanced)
        self.assertIn("{showInferenceSteps && (", advanced)
        self.assertIn("{showGuidanceScale && (", advanced)
        self.assertIn("Fast keeps its distilled CFG 1 recipe", advanced)
        self.assertIn(
            "!isAudio && !isScailEdit && <PostProcessing />",
            advanced,
        )
        self.assertIn(
            "(isAvatar && !isScailEdit)) && <WindowSettings />",
            advanced,
        )
        self.assertIn(
            "!isScailEdit && (modelOptions as Record<string, unknown> | null)?.perturbation",
            advanced,
        )
        self.assertIn(
            "!isScailEdit && (isVideo || isAvatar) && hasImageRefs",
            advanced,
        )
        self.assertIn("!isScailEdit && <div>", advanced)
        self.assertGreaterEqual(
            launch.count('"num_inference_steps": inference_steps'),
            2,
        )
        self.assertGreaterEqual(
            launch.count('"guidance_scale": guidance_scale'),
            2,
        )
        self.assertIn(
            "num_inference_steps: (state.params.num_inference_steps as number)",
            store,
        )
        self.assertIn(
            "repaintModel === 'scail2_14B'",
            store,
        )
        self.assertIn(
            "recastModel === 'scail2_14B'",
            store,
        )

    def test_recast_uses_native_replace_without_changing_studio_animate(self):
        launch = _read(_LAUNCH_PATH)
        scail2 = _read(_SCAIL2_PATH)
        store = _read(_STORE_PATH)
        self.assertIn('"scail2_recast_conditioning": "native_replace"', launch)
        self.assertIn("replace_mode = test_scail2_replace(video_prompt_type)", scail2)
        self.assertIn("replace_conditioning = test_scail2_replace(video_prompt_type)", scail2)
        self.assertNotIn("animation_recast", scail2)
        self.assertNotIn("_matte_ref_image", scail2)
        self.assertIn("m.model_type === 'scail2_14B_recast_fast'", store)
        self.assertIn("m.model_type === 'scail2_14B'", store)

    def test_recast_postprocess_preserves_and_recovers_resolved_seed(self):
        launch = _read(_LAUNCH_PATH)
        self.assertIn('"-map_metadata", "2"', launch)
        self.assertIn('file_sidecar["params"]["seed"] = resolved_seed', launch)
        self.assertIn("resolved_seed = _extract_output_seed(name)", launch)

    def test_aligned_reference_reuses_the_exact_target_mask(self):
        launch = _read(_LAUNCH_PATH)
        scail2 = _read(_SCAIL2_PATH)
        self.assertIn('"edit_recast_ref_aligned": reference_aligned_to_source', launch)
        self.assertIn('"scail2_reference_mask_path"', launch)
        self.assertIn("def _load_aligned_ref_mask", scail2)
        self.assertIn("ref_mask = _load_aligned_ref_mask(", scail2)

    def test_recast_ui_keeps_mapping_cards_and_moves_edited_frame_to_repaint(self):
        store = _read(_STORE_PATH)
        controls = _read(_RECAST_CONTROLS_PATH)
        repaint_controls = _read(_REPAINT_CONTROLS_PATH)
        self.assertIn("editRecastMappings: RecastCharacterMapping[]", store)
        self.assertIn("character_mappings: recastMappings.map", store)
        self.assertIn("additional_ref_image_paths: mapping.additionalRefs", store)
        self.assertIn("target.anchor === 'recast'", store)
        self.assertIn("const refUrl = api.getFileUrl(refName)", store)
        self.assertNotIn("sendFrameToImageMode('recast')", controls)
        self.assertIn("sendFrameToImageMode('repaint')", repaint_controls)
        self.assertIn("Characters ({mappings.length}/5)", controls)
        self.assertIn("Add character", controls)
        self.assertIn("More views", controls)
        self.assertIn("Prepared references", controls)
        self.assertIn("editRecastRefAligned: boolean", store)
        self.assertIn(
            "reference_aligned_to_source: mapping.referenceAlignedToSource",
            store,
        )
        self.assertNotIn("Reference is a full edited copy of the selected first frame", controls)
        self.assertIn("Edited first frame", repaint_controls)

    def test_recast_auto_face_detail_is_previewed_saved_and_automatic(self):
        launch = _read(_LAUNCH_PATH)
        store = _read(_STORE_PATH)
        controls = _read(_RECAST_CONTROLS_PATH)
        client = _read(os.path.join(
            _ROOT, "ui", "src", "api", "client.ts",
        ))

        self.assertIn('body.get("auto_face_detail") is not False', launch)
        self.assertIn(
            '"edit_recast_auto_face_detail": auto_face_detail',
            launch,
        )
        self.assertIn('"auto_face_detail_refs": auto_face_detail_refs', launch)
        self.assertIn("editRecastAutoFaceDetail: true", store)
        self.assertIn("auto_face_detail: true", store)
        self.assertIn(
            "p.edit_recast_auto_face_detail !== false",
            store,
        )
        self.assertNotIn("Automatically add a face-detail view", controls)
        self.assertIn("adds a face-detail view", controls)
        self.assertIn("'Face detail'", controls)
        self.assertIn("auto_face_detail?: boolean", client)
        self.assertIn("'auto_face_detail'", client)

    def test_recast_prompt_guidance_is_compatible_but_off_and_hidden(self):
        launch = _read(_LAUNCH_PATH)
        store = _read(_STORE_PATH)
        controls = _read(_RECAST_CONTROLS_PATH)
        self.assertIn(
            'body.get("enhance_prompt") is True',
            launch,
        )
        self.assertIn(
            '"edit_recast_enhance_prompt": enhance_prompt',
            launch,
        )
        self.assertIn(
            '"edit_recast_raw_prompt": raw_recast_prompt',
            launch,
        )
        self.assertIn("editRecastEnhancePrompt: false", store)
        self.assertIn("enhance_prompt: false", store)
        self.assertIn(
            "p.edit_recast_enhance_prompt === true",
            store,
        )
        self.assertIn(
            "typeof p.edit_recast_raw_prompt === 'string'",
            store,
        )
        self.assertNotIn("Add Maestro identity and scene guidance", controls)

    def test_repaint_reuses_scail_animate_and_is_a_first_class_edit_mode(self):
        launch = _read(_LAUNCH_PATH)
        store = _read(_STORE_PATH)
        controls = _read(_REPAINT_CONTROLS_PATH)
        toggle = _read(_EDIT_SUBMODE_PATH)
        client = _read(_API_CLIENT_PATH)

        self.assertIn('@api.post("/api/v1/repaint")', launch)
        self.assertIn('@api.post("/api/v1/repaint/preview")', launch)
        self.assertIn('model_type == "scail2_14B_fast"', launch)
        self.assertIn('"video_prompt_type": process_type', launch)
        self.assertIn('process_type = f"V{region_count}A"', launch)
        self.assertIn('"image_prompt_type": "S"', launch)
        self.assertIn('"image_start": prepared_target_path', launch)
        self.assertIn('"scail2_animate_preprocessing": "raw"', launch)
        self.assertIn('"audio_prompt_type": "R"', launch)
        self.assertIn('"input_video_strength": 1', launch)
        self.assertIn('"edit_sub_mode": "restyle"', launch)
        self.assertIn("submitRepaint({", store)
        self.assertIn("target_frame_path: state.editRepaintFramePath", store)
        self.assertIn("editRepaintMappings: RepaintRegionMapping[]", store)
        self.assertIn("target.anchor === 'repaint'", store)
        self.assertIn("submitRepaint", client)
        self.assertIn("repaintPreview", client)
        self.assertIn("{ value: 'restyle', label: 'Repaint' }", toggle)
        self.assertNotIn("{ value: 'restyle', label: 'Restyle', experimental: true }", toggle)
        self.assertIn("Fast is recommended (6 steps)", controls)
        self.assertIn("Track changed regions", controls)
        self.assertIn("source-to-edited-frame mapping", controls)

    def test_primary_edit_modes_use_the_product_order(self):
        toggle = _read(_EDIT_SUBMODE_PATH)
        ordered_modes = (
            "{ value: 'retake', label: 'Retake' }",
            "{ value: 'edit_anything', label: 'Edit Anything' }",
            "{ value: 'outpaint', label: 'Outpaint' }",
            "{ value: 'restyle', label: 'Repaint' }",
            "{ value: 'recast', label: 'Recast' }",
        )
        positions = [toggle.index(mode) for mode in ordered_modes]
        self.assertEqual(positions, sorted(positions))

    def test_recast_and_repaint_use_compact_copy_with_accessible_help(self):
        recast = _read(_RECAST_CONTROLS_PATH)
        repaint = _read(_REPAINT_CONTROLS_PATH)
        resolution = _read(_SCAIL_RESOLUTION_SELECTOR_PATH)
        tooltip = _read(_INFO_TOOLTIP_PATH)
        prompt = _read(_PROMPT_INPUT_PATH)

        for controls in (recast, repaint, resolution):
            self.assertIn("<InfoTooltip", controls)
        self.assertIn("createPortal", tooltip)
        self.assertIn('role="tooltip"', tooltip)
        self.assertIn("aria-describedby", tooltip)
        self.assertIn("aria-label={label}", tooltip)
        self.assertIn(
            "Describe the finished video and replacement characters...",
            prompt,
        )
        self.assertIn("editSubMode === 'restyle'", prompt)

        verbose_copy = (
            "Map each person in the source to one replacement character",
            "Maestro automatically isolates each reference",
            "Describe the finished video, not an instruction",
            "Repaint a video from an edited first frame",
            "Usually leave this empty for whole-frame Repaint",
            "Describe the finished video in the prompt below",
        )
        combined = recast + repaint
        for phrase in verbose_copy:
            self.assertNotIn(phrase, combined)

    def test_repaint_semantic_masks_use_native_opposite_backgrounds(self):
        launch = _read(_LAUNCH_PATH)
        self.assertIn(
            'background_color=(0, 0, 0)',
            launch,
        )
        self.assertIn(
            'background_color=(255, 255, 255)',
            launch,
        )
        self.assertIn(
            '"scail2_reference_mask_path"',
            launch,
        )
        self.assertIn(
            '"scail2_primary_reference_people"',
            launch,
        )


if __name__ == "__main__":
    unittest.main()
