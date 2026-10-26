"""Regression coverage for the pinned H3 LightX2V preset port."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
_MANIFEST_PATH = _APP / "models" / "minimax_h3" / "turbo_presets.json"
_TURBO_PATH = _APP / "models" / "minimax_h3" / "turbo.py"
_SCHEDULER_PATH = _APP / "models" / "minimax_h3" / "scheduler.py"
_MAIN_PATH = _APP / "models" / "minimax_h3" / "minimax_h3_main.py"
_LAUNCH_PATH = _APP / "launch.py"
_REVISION = "de1028e84b0fd5e7f15b2684b704d8c7289b02be"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_launch_function(name: str):
    import ast

    tree = ast.parse(_LAUNCH_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(_LAUNCH_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def _load_main_function(name: str, **namespace):
    import ast

    tree = ast.parse(_MAIN_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function_namespace = {"math": math, **namespace}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(_MAIN_PATH),
            "exec",
        ),
        function_namespace,
    )
    return function_namespace[name]


class H3LightX2VPresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.turbo = _load_module("maestro_h3_lightx2v_turbo_test", _TURBO_PATH)
        cls.manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.presets = {
            preset["id"]: preset for preset in cls.manifest["presets"]
        }
        cls.scheduler_module = _load_module(
            "maestro_h3_lightx2v_scheduler_test", _SCHEDULER_PATH
        )
        cls.set_generation_shifts = staticmethod(
            _load_main_function(
                "_set_h3_generation_shifts",
                minimax_h3_turbo_preset_for_path=(
                    cls.turbo.minimax_h3_turbo_preset_for_path
                ),
            )
        )

    def test_lightx2v_files_have_exact_pins_and_workflow_recipes(self):
        expected = {
            "lightx2v-fl2va-4step-v1.3-768p": (
                "minimax_h3_lightx2v_fl2v_turbo_4step_alpha8_v1.3_768p_bf16.safetensors",
                "fl2va",
                4,
                6.0,
                "25257cb9ebeae78f96101c3352a19fa487c5fa0434a4c33c88ae2c565e3b7de9",
            ),
            "lightx2v-fl2va-8step-v1.1-768p": (
                "minimax_h3_lightx2v_fl2v_turbo_8step_alpha8_v1.1_768p_bf16.safetensors",
                "fl2va",
                8,
                6.0,
                "6e2c616eb0adf03ab8eaa0113ec0c42ca0fa7073354eef01aa305fc7afb9861b",
            ),
            "lightx2v-ref2va-4step-v1.0-768p": (
                "minimax_h3_lightx2v_ref2v_turbo_4step_alpha8_v1.0_768p_bf16.safetensors",
                "ref2va",
                4,
                12.0,
                "92e8e2f987eea03955e5731df0baf1255460ccffc44ead94e05ca505f0e8579e",
            ),
            "lightx2v-ref2va-8step-v1.1-768p": (
                "minimax_h3_lightx2v_ref2v_turbo_8step_alpha8_v1.1_768p_bf16.safetensors",
                "ref2va",
                8,
                12.0,
                "e7493e07d90cb98798d34aba2afc0dcf24dd14304d83c8a9b95c3548f6ad8d30",
            ),
        }
        for preset_id, (filename, workflow, steps, shift, digest) in expected.items():
            with self.subTest(preset_id=preset_id):
                preset = self.presets[preset_id]
                self.assertEqual(preset["filename"], filename)
                self.assertEqual(
                    preset["remote_path"], f"loras/{filename}"
                )
                self.assertEqual(preset["repo_id"], "DeepBeepMeep/MiniMax-H3")
                self.assertEqual(preset["revision"], _REVISION)
                self.assertEqual(preset["sha256"], digest)
                self.assertEqual(preset["size"], 1_383_712_904)
                self.assertEqual(preset["workflow"], workflow)
                self.assertEqual(preset["runtime"], "standard_lora")
                self.assertEqual(preset["steps"], steps)
                self.assertEqual(preset["weight"], 1.0)
                self.assertEqual(
                    preset["generation_settings"],
                    {"guidance_scale": 1.0, "flow_shift": shift},
                )

    def test_selector_contains_only_matching_workflow_presets(self):
        expected_fl2va = {
            "lightx2v-fl2va-4step-v1.3-768p",
            "lightx2v-fl2va-8step-v1.1-768p",
        }
        expected_ref2va = {
            "lightx2v-ref2va-4step-v1.0-768p",
            "lightx2v-ref2va-8step-v1.1-768p",
        }
        for preset_id in expected_fl2va:
            self.assertIn(
                preset_id,
                {
                    item["id"]
                    for item in self.turbo.minimax_h3_turbo_presets_for_workflow(
                        "fl2va"
                    )
                },
            )
            self.assertNotIn(
                preset_id,
                {
                    item["id"]
                    for item in self.turbo.minimax_h3_turbo_presets_for_workflow(
                        "ref2va"
                    )
                },
            )
        for preset_id in expected_ref2va:
            self.assertIn(
                preset_id,
                {
                    item["id"]
                    for item in self.turbo.minimax_h3_turbo_presets_for_workflow(
                        "ref2va"
                    )
                },
            )
            self.assertNotIn(
                preset_id,
                {
                    item["id"]
                    for item in self.turbo.minimax_h3_turbo_presets_for_workflow(
                        "fl2va"
                    )
                },
            )

    def test_selected_preset_applies_shift_without_stacking_accelerators(self):
        standard_loras = {
            "lightx2v-fl2va-4step-v1.3-768p": (
                "fl2va",
                4,
                6.0,
                "minimax_h3_lightx2v_fl2v_turbo_4step_alpha8_v1.3_768p_bf16.safetensors",
            ),
            "lightx2v-fl2va-8step-v1.1-768p": (
                "fl2va",
                8,
                6.0,
                "minimax_h3_lightx2v_fl2v_turbo_8step_alpha8_v1.1_768p_bf16.safetensors",
            ),
            "lightx2v-ref2va-4step-v1.0-768p": (
                "ref2va",
                4,
                12.0,
                "minimax_h3_lightx2v_ref2v_turbo_4step_alpha8_v1.0_768p_bf16.safetensors",
            ),
            "lightx2v-ref2va-8step-v1.1-768p": (
                "ref2va",
                8,
                12.0,
                "minimax_h3_lightx2v_ref2v_turbo_8step_alpha8_v1.1_768p_bf16.safetensors",
            ),
        }
        for preset_id, (workflow, steps, shift, filename) in standard_loras.items():
            with self.subTest(preset_id=preset_id):
                body = {
                    "minimax_h3_turbo_mode": True,
                    "minimax_h3_turbo_preset": preset_id,
                    "activated_loras": [
                        "style.safetensors",
                        "MiniMax-H3-FL2VA-Acc-8Step.safetensors",
                    ],
                    "loras_multipliers": "0.75 1.0",
                    "num_inference_steps": 20,
                    "guidance_scale": 4.5,
                    "flow_shift": 99,
                }
                self.assertTrue(
                    self.turbo.normalize_minimax_h3_turbo_request(
                        body,
                        full_checkpoint=False,
                        workflow=workflow,
                    )
                )
                self.assertEqual(body["num_inference_steps"], steps)
                self.assertEqual(body["guidance_scale"], 1.0)
                self.assertEqual(body["flow_shift"], shift)
                self.assertEqual(
                    body["activated_loras"], ["style.safetensors", filename]
                )
                self.assertEqual(body["loras_multipliers"], "0.75 1.00")
                self.assertEqual(
                    self.turbo.find_minimax_h3_pdd_loras(
                        body["activated_loras"]
                    ),
                    [],
                )

    def test_specialized_h3_models_remain_outside_the_standard_selector(self):
        option = _load_launch_function("_minimax_h3_turbo_option")
        app_path = str(_APP)
        added_app_path = app_path not in sys.path
        if added_app_path:
            sys.path.insert(0, app_path)
        try:
            for model_def in (
                {"architecture": "minimax_h3", "minimax_h3_fused_turbo": True},
                {"architecture": "minimax_h3", "minimax_h3_fasth3": True},
                {"architecture": "minimax_h3", "vdn": True},
                {"architecture": "minimax_h3", "minimax_h3_baked_turbo": True},
            ):
                with self.subTest(model_def=model_def):
                    self.assertIsNone(option(model_def))
        finally:
            if added_app_path:
                sys.path.remove(app_path)

    def test_cpu_sigma_schedules_apply_recipe_shift_and_reset_between_jobs(self):
        scheduler_type = self.scheduler_module.MiniMaxH3Scheduler
        video = scheduler_type(shift=12.0)
        audio = scheduler_type(shift=3.0)
        lightx2v_filename = self.presets[
            "lightx2v-fl2va-4step-v1.3-768p"
        ]["filename"]

        # The preset's shift wins over WanGP's hidden generic fallback value.
        self.set_generation_shifts(
            video,
            audio,
            default_video_shift=12.0,
            default_audio_shift=3.0,
            model_def={"flow_shift": False},
            turbo_lora_paths=[lightx2v_filename],
            requested_video_shift=7.0,
        )
        video.set_timesteps(5, device="cpu")
        audio.set_timesteps(5, device="cpu")
        expected_lightx2v = scheduler_type(shift=6.0)
        expected_lightx2v.set_timesteps(5, device="cpu")
        expected_audio = scheduler_type(shift=3.0)
        expected_audio.set_timesteps(5, device="cpu")
        self.assertEqual(video.shift, 6.0)
        self.scheduler_module.torch.testing.assert_close(
            video.sigmas, expected_lightx2v.sigmas
        )
        self.assertEqual(audio.shift, 3.0)
        self.scheduler_module.torch.testing.assert_close(
            audio.sigmas, expected_audio.sigmas
        )

        # A following ordinary job has no recipe shift; it returns to the
        # checkpoint default instead of retaining the preceding shift 6.
        self.set_generation_shifts(
            video,
            audio,
            default_video_shift=12.0,
            default_audio_shift=3.0,
            model_def={"flow_shift": False},
            requested_video_shift=7.0,
        )
        video.set_timesteps(21, device="cpu")
        audio.set_timesteps(21, device="cpu")
        expected_default = scheduler_type(shift=12.0)
        expected_default.set_timesteps(21, device="cpu")
        expected_audio_default = scheduler_type(shift=3.0)
        expected_audio_default.set_timesteps(21, device="cpu")
        self.assertEqual(video.shift, 12.0)
        self.scheduler_module.torch.testing.assert_close(
            video.sigmas, expected_default.sigmas
        )
        self.assertEqual(audio.shift, 3.0)
        self.scheduler_module.torch.testing.assert_close(
            audio.sigmas, expected_audio_default.sigmas
        )

    def test_model_enabled_shift_and_fixed_recipe_policy(self):
        scheduler_type = self.scheduler_module.MiniMaxH3Scheduler
        video = scheduler_type(shift=12.0)
        audio = scheduler_type(shift=3.0)

        self.set_generation_shifts(
            video,
            audio,
            default_video_shift=12.0,
            default_audio_shift=3.0,
            model_def={"flow_shift": True},
            requested_video_shift=6.0,
        )
        video.set_timesteps(9, device="cpu")
        expected_explicit = scheduler_type(shift=6.0)
        expected_explicit.set_timesteps(9, device="cpu")
        self.scheduler_module.torch.testing.assert_close(
            video.sigmas, expected_explicit.sigmas
        )

        # A PDD/fixed recipe ignores a request shift while restoring its model
        # video/audio defaults before building the real schedule.
        self.set_generation_shifts(
            video,
            audio,
            default_video_shift=12.0,
            default_audio_shift=3.0,
            model_def={"flow_shift": True},
            requested_video_shift=6.0,
            allow_video_shift_override=False,
        )
        video.set_timesteps(9, device="cpu")
        audio.set_timesteps(9, device="cpu")
        expected_pdd_video = scheduler_type(shift=12.0)
        expected_pdd_video.set_timesteps(9, device="cpu")
        expected_pdd_audio = scheduler_type(shift=3.0)
        expected_pdd_audio.set_timesteps(9, device="cpu")
        self.assertEqual(video.shift, 12.0)
        self.scheduler_module.torch.testing.assert_close(
            video.sigmas, expected_pdd_video.sigmas
        )
        self.assertEqual(audio.shift, 3.0)
        self.scheduler_module.torch.testing.assert_close(
            audio.sigmas, expected_pdd_audio.sigmas
        )

        # Non-3 audio defaults (such as fixed DaSiWa recipes) remain model
        # owned too; they are restored alongside the video schedule.
        video = scheduler_type(shift=9.0)
        audio = scheduler_type(shift=4.0)
        self.set_generation_shifts(
            video,
            audio,
            default_video_shift=9.0,
            default_audio_shift=4.0,
            model_def={"flow_shift": True},
            requested_video_shift=6.0,
            allow_video_shift_override=False,
        )
        video.set_timesteps(5, device="cpu")
        audio.set_timesteps(5, device="cpu")
        expected_dasiwa_video = scheduler_type(shift=9.0)
        expected_dasiwa_video.set_timesteps(5, device="cpu")
        expected_dasiwa_audio = scheduler_type(shift=4.0)
        expected_dasiwa_audio.set_timesteps(5, device="cpu")
        self.assertEqual(video.shift, 9.0)
        self.scheduler_module.torch.testing.assert_close(
            video.sigmas, expected_dasiwa_video.sigmas
        )
        self.assertEqual(audio.shift, 4.0)
        self.scheduler_module.torch.testing.assert_close(
            audio.sigmas, expected_dasiwa_audio.sigmas
        )


if __name__ == "__main__":
    unittest.main()
