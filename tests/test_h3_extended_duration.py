"""Exercise extended-pass geometry without loading models or using the GPU."""
import ast
from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
from models.minimax_h3.duration import h3_duration_model_def, apply_h3_duration_override
from models.minimax_h3.minimax_h3_handler import family_handler
from models.minimax_h3.packing import align_num_frames
from services.studio_enhancement import enhancement_request, new_enhancement
from services.perf_recommend import compute_h3_weight_budget


class H3ExtendedDurationTests(unittest.TestCase):
    @staticmethod
    def model(mode="minimax_h3"):
        # WGP merges the handler's capabilities into the checkpoint definition.
        return {**family_handler.query_model_def(mode, {}), "architecture": mode}

    def test_opt_in_only_and_no_shared_model_mutation(self):
        model = self.model()
        original = deepcopy(model)
        for flag in (None, False, "true", 1):
            self.assertIs(h3_duration_model_def(model, {"minimax_h3_extended_duration": flag}), model)
        params = {"minimax_h3_extended_duration": True}
        extended = apply_h3_duration_override(params, model)
        self.assertEqual(extended["frames_maximum"], 719)
        self.assertEqual(extended["sliding_window_defaults"]["window_max"], 719)
        self.assertTrue(params["sliding_window_memory_override"])
        self.assertEqual(model, original)
        for excluded in ({**model, "audio_only": True}, {**model, "minimax_h3_viggle": True},
                         {"architecture": "ltx2", "frames_maximum": 257}):
            self.assertIs(h3_duration_model_def(excluded, params), excluded)

    def test_frames_and_references_survive_validation_and_restore_as_one_pass(self):
        for mode in ("minimax_h3", "minimax_h3_ref2va", "minimax_h3_full", "minimax_h3_ref2va_full"):
            with self.subTest(mode=mode):
                model = self.model(mode)
                params = {"model_type": mode, "prompt": "A quiet garden at sunrise.",
                          "video_length": 720, "sliding_window_size": 719,
                          "sliding_window_overlap": 18, "resolution": "864x480",
                          "minimax_h3_extended_duration": True}
                with patch("torch.cuda.is_available", return_value=False):
                    self.assertIsNone(family_handler.validate_generative_settings(mode, model, params))
                self.assertEqual(params["video_length"], 719)
                self.assertEqual(params["sliding_window_size"], 719)
                self.assertFalse(params.get("minimax_h3_multi_window", False))
                self.assertFalse(params.get("minimax_h3_reference_sequence", False))
                restored = deepcopy(params)
                family_handler.fix_settings(mode, 2.58, model, restored)
                self.assertEqual(restored["video_length"], 719)
                self.assertEqual(restored["sliding_window_size"], 719)
                normal = {**params, "minimax_h3_extended_duration": False}
                with patch("torch.cuda.is_available", return_value=False):
                    self.assertIsNone(family_handler.validate_generative_settings(mode, model, normal))
                self.assertEqual(normal["video_length"], 345)
                self.assertEqual(normal["sliding_window_size"], 345)

    def test_wgp_native_alignment_keeps_extended_pass_whole(self):
        # Importing wgp starts the application; execute its real pure geometry helpers.
        source = ast.parse((APP / "wgp.py").read_text(encoding="utf-8"))
        names = {"align_model_frame_count", "normalize_model_total_frame_count", "compute_next_sliding_window_length"}
        namespace = {}
        exec(compile(ast.Module(body=[n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names],
                                type_ignores=[]), "wgp.py", "exec"), namespace)
        model = h3_duration_model_def(self.model(),
                                      {"minimax_h3_extended_duration": True})
        frames = namespace["normalize_model_total_frame_count"](719, model, window_size=719)
        window = namespace["align_model_frame_count"](719, model)
        self.assertEqual((frames, window), (719, 719))
        self.assertFalse(frames > window, "WGP must not dispatch a continuation pass")

    def test_runtime_duration_guard_requires_explicit_opt_in(self):
        # Stop at the next validation (steps=1) before any tensor allocation.
        source = ast.parse((APP / "models/minimax_h3/minimax_h3_main.py").read_text(encoding="utf-8"))
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3Model")
        generate = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
        generate.decorator_list = []
        namespace = {"__package__": "models.minimax_h3", "MINIMAX_H3_FPS": 24,
                     "MINIMAX_H3_MIN_DURATION": 5, "MINIMAX_H3_MAX_DURATION": 15,
                     "math": math, "align_num_frames": align_num_frames}
        exec(compile(ast.Module(body=[generate], type_ignores=[]), "h3-generate.py", "exec"), namespace)
        model = SimpleNamespace(audio_only=False, viggle=False, _pdd_lora_active=False)
        for flag, frames, message in ((False, 719, "duration limit"), (True, 719, "scheduler grid"),
                                     (True, 736, "duration limit"), (False, 345, "scheduler grid")):
            with self.subTest(flag=flag, frames=frames), self.assertRaisesRegex(ValueError, message):
                namespace["generate"](model, "A garden.", frame_num=frames, sampling_steps=1,
                                      minimax_h3_extended_duration=flag)

    def test_queue_and_enhancer_keep_the_full_duration_and_opt_in(self):
        model = self.model()
        params = {"prompt": "A sunrise.", "model_type": "minimax_h3", "video_length": 719,
                  "sliding_window_size": 719, "minimax_h3_extended_duration": True}
        saved = new_enhancement(params, {})["original_params"]
        self.assertTrue(saved["minimax_h3_extended_duration"])
        payload, sequence = enhancement_request(saved, model)
        self.assertFalse(sequence)
        self.assertEqual(payload["window_count"], 1)
        self.assertEqual(payload["duration_seconds"], 719 / 24)
        self.assertEqual(payload["window_size_seconds"], 719 / 24)

    def test_memory_budget_accounts_for_longer_pass(self):
        normal = compute_h3_weight_budget(24, "864x480", 345)
        extended = compute_h3_weight_budget(24, "864x480", 719)
        self.assertGreater(extended["compute_ratio"], normal["compute_ratio"])
        self.assertGreater(extended["activation_reserve_gb"], normal["activation_reserve_gb"])
        self.assertLess(extended["weight_budget_gb"], normal["weight_budget_gb"])


if __name__ == "__main__":
    unittest.main()
