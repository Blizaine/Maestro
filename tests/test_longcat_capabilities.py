"""Lightweight LongCat capability metadata regressions (no model runtime)."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


_ROOT = Path(__file__).resolve().parents[1]
_HANDLER_PATH = _ROOT / "app" / "models" / "longcat" / "longcat_handler.py"
_MAIN_PATH = _ROOT / "app" / "models" / "longcat" / "longcat_main.py"
_LAUNCH_PATH = _ROOT / "app" / "launch.py"
_DEFAULTS_PATH = _ROOT / "app" / "defaults"


def _load_handler():
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = object()
    torch_stub.float32 = object()

    shared_stub = types.ModuleType("shared")
    shared_stub.__path__ = []
    utils_stub = types.ModuleType("shared.utils")
    utils_stub.__path__ = []
    hf_stub = types.ModuleType("shared.utils.hf")
    hf_stub.build_hf_url = lambda repo, folder, filename: f"{repo}/{folder}/{filename}"

    modules = {
        "torch": torch_stub,
        "shared": shared_stub,
        "shared.utils": utils_stub,
        "shared.utils.hf": hf_stub,
    }
    spec = importlib.util.spec_from_file_location("longcat_handler_test", _HANDLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class LongCatCapabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler = _load_handler().family_handler

    def test_video_declares_text_and_image_generation_without_audio_input(self):
        model_def = self.handler.query_model_def("longcat_video", {})

        self.assertTrue(model_def["t2v_class"])
        self.assertTrue(model_def["i2v_class"])
        self.assertFalse(model_def.get("any_audio_prompt", False))
        self.assertNotIn("audio_prompt_type_sources", model_def)

    def test_each_model_exposes_the_existing_short_window_recipe_to_studio(self):
        for architecture in ("longcat_video", "longcat_avatar"):
            with self.subTest(architecture=architecture):
                model_def = self.handler.query_model_def(architecture, {})
                windows = model_def["sliding_window_defaults"]
                self.assertEqual(windows["window_default"], 93)
                self.assertEqual(windows["window_max"], 93)
                self.assertEqual(windows["window_step"], 4)
                self.assertEqual(windows["overlap_default"], 13)
                self.assertEqual(windows["discard_last_frames"], 0)

    def test_oversized_or_inherited_windows_are_bounded_without_cutting_the_song(self):
        normalize = _load_handler().normalize_longcat_window_params
        for model_type in ("longcat_video", "longcat_avatar", "longcat_avatar_multi"):
            body = {"model_type": model_type, "video_length": 1275, "duration_seconds": 79.6875,
                    "resolution": "1280x720", "sliding_window_size": 640,
                    "sliding_window_overlap": 18, "sliding_window_discard_last_frames": 8}
            self.assertTrue(normalize(body, {}))
            self.assertEqual(body["sliding_window_size"], 93)
            self.assertEqual(body["sliding_window_overlap"], 13)
            self.assertEqual(body["sliding_window_discard_last_frames"], 0)
            self.assertEqual(body["video_length"], 1275)
            self.assertEqual(body["duration_seconds"], 79.6875)
            self.assertEqual(body["resolution"], "1280x720")
            self.assertFalse(normalize(body, {}), "normalization is idempotent")
        unrelated = {"model_type": "ltx2_22B", "video_length": 1275, "sliding_window_size": 640}
        self.assertFalse(normalize(unrelated, {"architecture": "ltx2"}))
        self.assertEqual(unrelated["sliding_window_size"], 640)

    def test_longcat_window_rounding_defaults_and_invalid_geometry(self):
        normalize = _load_handler().normalize_longcat_window_params
        for requested, expected in ((1, 17), (17, 17), (23, 25), (93, 93), (641, 93)):
            body = {"model_type": "longcat_avatar", "sliding_window_size": requested}
            normalize(body, {})
            self.assertEqual(body["sliding_window_size"], expected)
            self.assertLess(body["sliding_window_overlap"], body["sliding_window_size"])
        body = {"model_type": "longcat_avatar"}
        normalize(body, {})
        self.assertEqual(body["sliding_window_size"], 93)
        for invalid in (True, -1, 0, 1.5, "bad", float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize({"model_type": "longcat_avatar", "sliding_window_size": invalid}, {})

    def test_api_and_restored_queue_bound_windows_before_model_work(self):
        # Execute the production request and worker boundary guards. Restored
        # held jobs may enter the worker without going through the request API.
        launch = ast.parse(_LAUNCH_PATH.read_text(encoding="utf-8"))
        handler = _load_handler()
        for function_name in ("_prepare_generation_submission", "_run_generation"):
            function = next(node for node in launch.body
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name == function_name)
            guard = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                         and any(isinstance(part, ast.ImportFrom)
                                 and part.module == "models.longcat.longcat_handler"
                                 for part in node.body))
            code = compile(ast.Module(body=[guard], type_ignores=[]), str(_LAUNCH_PATH), "exec")
            for model_type, architecture in (("longcat_avatar", "longcat_avatar"),
                                              ("longcat_avatar_multi", "longcat_avatar"),
                                              ("longcat_video", "longcat_video"),
                                              ("ltx2_22B", "ltx2")):
                with self.subTest(boundary=function_name, model_type=model_type):
                    body = {"model_type": model_type, "video_length": 1275,
                            "sliding_window_size": 640, "sliding_window_overlap": 18}
                    definition = {"architecture": architecture}
                    namespace = {"body": body, "_runtime_params": body,
                                 "_base_model_type": architecture,
                                 "_generation_model_def": definition,
                                 "_runtime_model_def": definition,
                                 "print": lambda *args: None}
                    with mock.patch.dict(sys.modules, {"models.longcat.longcat_handler": handler}):
                        exec(code, namespace)
                    self.assertEqual(body["video_length"], 1275)
                    if model_type.startswith("longcat"):
                        self.assertEqual(body["sliding_window_size"], 93)
                        self.assertEqual(body["sliding_window_overlap"], 13)
                    else:
                        self.assertEqual(body["sliding_window_size"], 640)

    def test_avatar_weight_budget_scales_with_the_per_pass_token_grid(self):
        budget = _load_handler().longcat_avatar_weight_budget
        at_720p = budget(24, "1280x720", 93)
        self.assertEqual(at_720p["window_tokens"], 86400)
        self.assertEqual(at_720p["weight_budget_gb"], 7)
        self.assertEqual(at_720p["activation_reserve_gb"], 17)
        self.assertFalse(at_720p["activation_reserve_clamped"])
        at_480p = budget(24, "832x480", 93)
        self.assertGreater(at_480p["weight_budget_gb"], at_720p["weight_budget_gb"])
        shorter = budget(24, "1280x720", 45)
        self.assertGreater(shorter["weight_budget_gb"], at_720p["weight_budget_gb"])
        self.assertEqual(budget(24, "1280x720", 93, 1)["weight_budget_gb"], 6)
        continuation = budget(24, "1280x720", 93, continuation=True)
        self.assertEqual(continuation["window_tokens"], 90000)
        self.assertEqual(continuation["reference_latent_frames"], 1)
        self.assertEqual(continuation["weight_budget_gb"], 3.5)
        self.assertGreater(continuation["requested_activation_reserve_gb"],
                           at_720p["requested_activation_reserve_gb"])
        self.assertGreater(budget(24, "832x480", 93, continuation=True)["weight_budget_gb"],
                           continuation["weight_budget_gb"])
        tight_card = budget(12, "1280x720", 93)
        self.assertTrue(tight_card["activation_reserve_clamped"])
        self.assertEqual(tight_card["weight_budget_gb"], 3.5)

    def test_avatar_worker_applies_streaming_budget_reloads_and_restores_settings(self):
        launch = ast.parse(_LAUNCH_PATH.read_text(encoding="utf-8"))
        names = {"_stage_count_from_params", "_apply_per_job_coefficient", "_restore_base_coefficient"}
        functions = [node for node in launch.body if isinstance(node, ast.FunctionDef) and node.name in names]
        handler = _load_handler()
        policy_path = _ROOT / "app/services/perf_recommend.py"
        spec = importlib.util.spec_from_file_location("longcat_perf_test", policy_path)
        policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy)
        engine = types.SimpleNamespace(
            args=types.SimpleNamespace(vram_safety_coefficient=0.8, transformer_budget=0),
            server_config={"vram_safety_coefficient": 0.8},
            get_model_def=lambda _: {"architecture": "longcat_avatar"},
            get_base_model_type=lambda _: "longcat_avatar", get_lora_dir=lambda _: None,
            wan_model=types.SimpleNamespace(_maestro_profile_vram_coefficient=0.8,
                                          _maestro_profile_transformer_budget_override_mb=0),
            reload_needed=False,
        )
        namespace = {"wgp": engine, "os": os, "_get_cached_hardware": lambda: {"gpu_vram_gb": 24},
                     "_BASE_TRANSFORMER_BUDGET_MB": None, "_H3_RESIDENCY_HEADROOM": 0.97,
                     "print": lambda *args: None}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(_LAUNCH_PATH), "exec"), namespace)
        job = {"params": {"model_type": "longcat_avatar", "resolution": "1280x720",
                          "video_length": 1275, "sliding_window_size": 93}}
        with mock.patch.dict(sys.modules, {"services.perf_recommend": policy,
                                          "models.longcat.longcat_handler": handler}):
            namespace["_apply_per_job_coefficient"](job)
        self.assertIn("vram_adjustment", job, "worker must not silently skip Avatar budgeting")
        self.assertEqual(job["vram_adjustment"]["longcat_avatar_budget"]["window_tokens"], 90000)
        self.assertAlmostEqual(engine.args.vram_safety_coefficient, 3.5 / 24)
        self.assertEqual(engine.args.transformer_budget, int(3.5 * 1024 * 0.97))
        self.assertTrue(engine.reload_needed, "cached full-residency model cannot retain its old hooks")
        self.assertEqual(engine.server_config["vram_safety_coefficient"], 0.8)
        namespace["_restore_base_coefficient"]()
        self.assertEqual(engine.args.vram_safety_coefficient, 0.8)
        self.assertEqual(engine.args.transformer_budget, 0)
        self.assertIsNone(namespace["_BASE_TRANSFORMER_BUDGET_MB"])

    def test_full_residency_profiles_honor_only_the_temporary_transformer_budget(self):
        wgp_path = _ROOT / "app/wgp.py"
        tree = ast.parse(wgp_path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "init_pipe")
        args = types.SimpleNamespace(preload=0, transformer_budget=6952)
        namespace = {"args": args, "server_config": {}}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(wgp_path), "exec"), namespace)
        for profile in (1, 3):
            settings = {}
            namespace["init_pipe"]({"transformer": object()}, settings, profile)
            self.assertEqual(settings["budgets"]["transformer"], 6952)
            self.assertNotIn("text_encoder", settings["budgets"])
        args.transformer_budget = 0
        settings = {}
        namespace["init_pipe"]({"transformer": object()}, settings, 1)
        self.assertNotIn("transformer", settings["budgets"])
        args.transformer_budget, args.preload = 6952, 500
        settings = {}
        namespace["init_pipe"]({"transformer": object()}, settings, 1)
        self.assertNotIn("transformer", settings["budgets"], "explicit preload keeps its existing contract")

    def test_each_avatar_checkpoint_declares_anchor_image_and_audio_capabilities(self):
        for model_type in ("longcat_avatar", "longcat_avatar_multi"):
            with self.subTest(model_type=model_type):
                source = json.loads(
                    (_DEFAULTS_PATH / f"{model_type}.json").read_text(encoding="utf-8")
                )["model"]
                model_def = self.handler.query_model_def(source["architecture"], source)

                self.assertTrue(model_def["t2v_class"])
                self.assertTrue(model_def["i2v_class"])
                self.assertTrue(model_def["any_audio_prompt"])
                self.assertTrue(model_def["audio_prompt_choices"])
                self.assertEqual(model_def["max_image_refs"], 1)
                self.assertEqual(
                    model_def["image_ref_choices"]["choices"],
                    [("None", ""), ("Anchor Reference Image", "KI")],
                )
                if model_type == "longcat_avatar_multi":
                    self.assertNotIn("audio_prompt_type_sources", model_def)
                    self.assertTrue(source["multi_speakers_only"])
                else:
                    self.assertEqual(
                        model_def["audio_prompt_type_sources"]["selection"],
                        ["A"],
                    )
                    self.assertEqual(
                        model_def["audio_prompt_type_sources"]["default"],
                        "A",
                    )

    def test_multi_speaker_branch_builds_both_guides_and_requires_second_audio(self):
        source = _MAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(_MAIN_PATH))
        helpers = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {
                "_is_multi_speaker_avatar",
                "_build_avatar_audio_embeddings",
            }
        ]
        namespace = {}
        exec(
            compile(
                ast.Module(body=helpers, type_ignores=[]),
                str(_MAIN_PATH),
                "exec",
            ),
            namespace,
        )
        is_multi_avatar = namespace["_is_multi_speaker_avatar"]
        build_embeddings = namespace["_build_avatar_audio_embeddings"]

        self.assertFalse(is_multi_avatar("longcat_avatar"))
        self.assertTrue(is_multi_avatar("longcat_avatar_multi"))
        self.assertIn("self.model_type = model_type", source)
        self.assertIn("audio_emb = _build_avatar_audio_embeddings(", source)
        self.assertIn("model_type=model_type", _HANDLER_PATH.read_text(encoding="utf-8"))

        calls = []

        def fake_audio_window_builder(audio_path, frame_num, fps, start_frame, stride):
            calls.append((audio_path, frame_num, fps, start_frame, stride))
            return (f"embedding:{audio_path}",)

        def fake_concatenate(first, second):
            return first + second

        with self.assertRaisesRegex(ValueError, "Second audio guide is required"):
            build_embeddings(
                "longcat_avatar_multi",
                "speaker-a.wav",
                None,
                fake_audio_window_builder,
                97,
                24,
                5,
                fake_concatenate,
            )
        self.assertEqual(calls, [], "validate both guides before building costly audio windows")

        multi_embeddings = build_embeddings(
            "longcat_avatar_multi",
            "speaker-a.wav",
            "speaker-b.wav",
            fake_audio_window_builder,
            97,
            24,
            5,
            fake_concatenate,
        )
        self.assertEqual(
            calls,
            [
                ("speaker-a.wav", 97, 24, 5, 2),
                ("speaker-b.wav", 97, 24, 5, 2),
            ],
        )
        self.assertEqual(multi_embeddings, ("embedding:speaker-a.wav", "embedding:speaker-b.wav"))

        calls.clear()
        single_embeddings = build_embeddings(
            "longcat_avatar",
            "speaker-a.wav",
            "unused.wav",
            fake_audio_window_builder,
            97,
            24,
            5,
            fake_concatenate,
        )
        self.assertEqual(calls, [("speaker-a.wav", 97, 24, 5, 2)])
        self.assertEqual(single_embeddings, ("embedding:speaker-a.wav",))

    def test_speaker_boxes_preserve_voice_order_and_horizontal_vertical_axes(self):
        # Run the production parser and mask builder with CPU arrays, without
        # importing the heavy LongCat pipeline or loading model weights.
        import numpy as np

        parser_path = _ROOT / "app" / "models" / "wan" / "multitalk" / "multitalk.py"
        parser_tree = ast.parse(parser_path.read_text(encoding="utf-8"))
        parser = next(node for node in parser_tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "parse_speakers_locations")
        main_tree = ast.parse(_MAIN_PATH.read_text(encoding="utf-8"))
        builder = next(node for node in ast.walk(main_tree) if isinstance(node, ast.FunctionDef)
                       and node.name == "_build_ref_target_masks")
        torch_arrays = types.SimpleNamespace(
            zeros=np.zeros, tensor=lambda values, **_kwargs: np.array(values), where=np.where,
            stack=lambda values, dim: np.stack(values, axis=dim),
        )
        namespace = {"torch": torch_arrays}
        exec(compile(ast.Module(body=[parser, builder], type_ignores=[]), str(_MAIN_PATH), "exec"), namespace)
        boxes, error = namespace["parse_speakers_locations"]("10:20:40:80 60:10:90:70")
        self.assertEqual(error, "")
        masks = namespace["_build_ref_target_masks"](None, 80, 200, boxes)
        expected = np.zeros((3, 80, 200))
        expected[0, 16:64, 20:80] = 1
        expected[1, 8:56, 120:180] = 1
        expected[2] = (expected[0] + expected[1] == 0)
        np.testing.assert_array_equal(masks, expected)
        # Legacy left:right recipes still match the ordered default boxes.
        boxes, error = namespace["parse_speakers_locations"]("0:50 50:100")
        self.assertEqual(error, "")
        defaults = namespace["_build_ref_target_masks"](None, 100, 200, boxes)
        self.assertEqual(defaults[0, 50, 60], 1)
        self.assertEqual(defaults[0, 50, 140], 0)
        self.assertEqual(defaults[1, 50, 140], 1)
        self.assertEqual(defaults[2, 0, 0], 1)


if __name__ == "__main__":
    unittest.main()
