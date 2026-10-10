"""Bounded, stdlib-only H3 per-job split policy tests."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.h3_performance_policy import (  # noqa: E402
    _policy,
    _resolution_area,
    h3_performance_advice,
    largest_actual_h3_pass,
)
from services.performance_tuning import build_performance_plan, workload_key  # noqa: E402


def hardware(**changes):
    value = {
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 3080 Ti",
        "gpu_vram_gb": 12.0,
        "ram_gb": 31.76,
        "allocator_active": "vmm",
        "platform": "win32",
    }
    value.update(changes)
    return value


def config(**changes):
    value = {
        "video_profile": 4,
        "transformer_quantization": "int8",
        "vram_allocator": "vmm",
        "smart_memory_pinning": True,
        "read_ahead": False,
        "attention_head_split": 0,
        "services": {
            "auto_performance": True,
            "auto_performance_defaults": {"attention_head_split": 0},
        },
    }
    value.update(changes)
    return value


def model(*, fused=True, **changes):
    value = {
        "architecture": "minimax_h3",
        "minimax_h3_full_checkpoint": False,
        "minimax_h3_fused_turbo": fused,
        "checkpoint_name": "MiniMax-H3-int8_convrot.safetensors",
        "URLs": ["https://example.invalid/minimax_h3_int8_convrot.safetensors"],
        "sla_attention_default": fused,
        "auto_resolution_budgets": {
            "auto_720p": 1280 * 704,
            "auto_1080p": 1920 * 1088,
        },
    }
    value.update(changes)
    return value


def params(frames=362, resolution="1280x704", *, fused=True, **changes):
    value = {
        "model_type": "minimax_h3_fused_turbo" if fused else "minimax_h3",
        "resolution": resolution,
        "video_length": frames,
        "sliding_window_size": frames,
        "num_inference_steps": 6 if fused else 20,
        "override_attention": "sla" if fused else "sage2",
        "minimax_h3_text_encoder": "gguf_q2_k",
    }
    value.update(changes)
    return value


class H3PolicyTests(unittest.TestCase):
    def advise(self, *, frames=362, resolution="1280x704", fused=True,
               hw=None, cfg=None, model_def=None, **request_changes):
        return h3_performance_advice(
            hw or hardware(),
            cfg or config(),
            model_def or model(fused=fused),
            params(frames, resolution, fused=fused, **request_changes),
        )

    def test_benchmark_bands_follow_completed_workload_points(self):
        cases = (
            (362, "1280x704", 0, 95_366),
            (124, "1920x1088", 0, 75_894),
            (243, "1920x1088", 1, 147_690),
            (362, "1920x1088", 2, 219_486),
        )
        for frames, resolution, split, target_rows in cases:
            with self.subTest(frames=frames, resolution=resolution):
                advice = self.advise(frames=frames, resolution=resolution)
                self.assertEqual(advice["attention_head_split"], split)
                self.assertEqual(advice["target_rows"], target_rows)
                self.assertEqual(advice["split_label"], ("Off", "Low", "Medium")[split])
                self.assertEqual(advice["risk"], "measured")

    def test_smaller_canvas_recomputes_to_off_without_mutating_saved_config(self):
        saved = config(attention_head_split=2)
        before = copy.deepcopy(saved)
        advice = h3_performance_advice(
            hardware(), saved, model(),
            params(362, "1280x704"),
        )
        self.assertEqual(advice["attention_head_split"], 0)
        self.assertEqual(saved, before)

    def test_scope_excludes_images_non_h3_full_audio_and_out_of_band_hardware(self):
        candidates = (
            (model(), params(image_mode=True), hardware()),
            ({"architecture": "other", "minimax_h3_full_checkpoint": False}, params(), hardware()),
            (model(minimax_h3_full_checkpoint=True), params(), hardware()),
            (model(architecture="minimax_h3_voice_audio"), params(), hardware()),
            (model(minimax_h3_viggle=True), params(), hardware()),
            (model(), params(output_type="image"), hardware()),
            (model(), params(), hardware(gpu_vram_gb=10.0)),
            (model(), params(), hardware(gpu_vram_gb=13.0)),
            (model(), params(), hardware(ram_gb=29.9)),
        )
        for definition, request, hw in candidates:
            with self.subTest(architecture=definition.get("architecture"), request=request, hw=hw):
                self.assertIsNone(h3_performance_advice(hw, config(), definition, request))

    def test_flat_budget_presets_explicit_portrait_and_alignment_are_resolved(self):
        definition = {
            "auto_resolution_budgets": {"auto_1080p": 1920 * 1088},
            "omni_sequence_memory_policy": {
                "auto_resolution_pixels": {"auto_portrait": 1088 * 1920},
            },
        }
        policy = _policy()
        geometry = policy["geometry"]
        preset = _resolution_area({"resolution": "auto_1080p"}, definition, geometry)
        portrait = _resolution_area({"resolution": "704x1280"}, definition, geometry)
        rounded = _resolution_area({"resolution": "1920x1080"}, definition, geometry)
        nested_preset = _resolution_area({"resolution": "auto_portrait"}, definition, geometry)
        self.assertEqual(preset["pixels"], 1920 * 1088)
        self.assertEqual(portrait["pixels"], 704 * 1280)
        self.assertEqual(rounded["pixels"], 1920 * 1088)
        self.assertTrue(rounded["rounded"])
        self.assertEqual(nested_preset["pixels"], 1920 * 1088)

    def test_actual_pass_uses_window_or_full_single_window_and_sequential_clip_schedule(self):
        self.assertEqual(largest_actual_h3_pass({
            "video_length": 362, "sliding_window_size": 243,
            "minimax_h3_multi_window": True,
        }), 243)
        self.assertEqual(largest_actual_h3_pass({
            "video_length": 362, "sliding_window_size": 124,
            "minimax_h3_multi_window": False,
        }), 362)
        self.assertIsNone(largest_actual_h3_pass({
            "video_length": 362, "sliding_window_size": "invalid",
            "minimax_h3_multi_window": True,
        }))
        self.assertEqual(largest_actual_h3_pass({
            "video_length": 729, "sliding_window_size": 124,
            "minimax_h3_multi_window": False,
            "multi_prompts_gen_type": 3,
            "per_clip_frames": [243, 243, 243],
        }), 243)

    def test_unknown_geometry_and_invalid_canvas_keep_saved_level(self):
        for request in (
            {"video_length": 362, "resolution": "unknown-preset"},
            {"video_length": "invalid", "resolution": "1920x1088"},
            {"video_length": 4, "resolution": "1920x1088"},
        ):
            request["model_type"] = "minimax_h3_fused_turbo"
            advice = h3_performance_advice(hardware(), config(), model(), request)
            self.assertIsNone(advice["attention_head_split"])
            self.assertEqual(advice["risk"], "caution")
            self.assertTrue(advice["warnings"])

    def test_lora_and_reference_uncertainty_are_explicit(self):
        with_lora = self.advise(activated_loras=["style.safetensors"])
        self.assertTrue(any("LoRAs were not included" in item for item in with_lora["warnings"]))
        with_reference = self.advise(scene_references=["reference.png"])
        self.assertTrue(any("References and guides add unmeasured" in item for item in with_reference["warnings"]))
        self.assertNotEqual(with_lora["risk"], "measured")
        self.assertNotEqual(with_reference["risk"], "measured")
        self.assertFalse(any("no exact completed benchmark point" in item
                             for item in with_reference["warnings"]))

    def test_checkpoint_file_identity_can_be_checked_without_exposing_paths(self):
        definition = model(checkpoint_name=None, URLs=[], _checkpoint_files=["C:/private/model_int8_convrot.safetensors"])
        advice = h3_performance_advice(hardware(), config(), definition, params(362, "1280x704"))
        self.assertEqual(advice["risk"], "measured")
        self.assertFalse(any("C:/private" in item for item in advice["warnings"] + advice["reasons"]))

    def test_pruned_long_canvas_warns_about_slow_observed_runs_without_estimating_duration(self):
        advice = self.advise(
            frames=345, resolution="1920x1088", fused=False,
            model_def=model(fused=False),
        )
        self.assertEqual(advice["risk"], "slow")
        slow = next(item for item in advice["warnings"] if "Long Pruned20" in item)
        self.assertIn("about 116 minutes", slow)
        self.assertIn("not its completion-time estimate", slow)

    def test_recipe_differences_and_active_allocator_are_unverified(self):
        wrong_encoder = self.advise(minimax_h3_text_encoder="gguf_q5_k")
        self.assertTrue(any("Q2K text encoder" in item for item in wrong_encoder["warnings"]))
        self.assertNotEqual(wrong_encoder["risk"], "measured")

        active_default = self.advise(hw=hardware(allocator_active="default"))
        self.assertTrue(any("active VMM allocator" in item for item in active_default["warnings"]))
        self.assertNotEqual(active_default["risk"], "measured")

        no_allocator_telemetry = self.advise(hw=hardware(allocator_active=None))
        self.assertTrue(any("pending restart" in item for item in no_allocator_telemetry["warnings"]))
        self.assertNotEqual(no_allocator_telemetry["risk"], "measured")

    def test_selected_checkpoint_wins_over_template_quantization_metadata(self):
        actual_gguf = self.advise(model_def=model(_checkpoint_files="selected-Q4_K_M.gguf"))
        self.assertNotEqual(actual_gguf["risk"], "measured")
        self.assertTrue(any("checkpoint identity" in warning for warning in actual_gguf["warnings"]))
        actual_int8 = self.advise(model_def=model(
            _checkpoint_files=["selected-int8_convrot.safetensors"],
            checkpoint_name="generic.safetensors", URLs=["other-quantization.gguf"],
        ))
        self.assertEqual(actual_int8["risk"], "measured")

    def test_shared_json_has_numeric_target_thresholds_and_encoder_evidence(self):
        policy = json.loads((ROOT / "app/shared/h3_performance_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(policy["bands"]["off_max_target_rows"], 100000)
        self.assertEqual(policy["bands"]["low_max_target_rows"], 165000)
        self.assertEqual(policy["benchmark"]["text_encoder"], "gguf_q2_k")
        self.assertEqual(policy["benchmark"]["variants"]["fused_turbo"]["steps"], 6)


class H3PlannerIntegrationTests(unittest.TestCase):
    def make_plan(self, *, hw=None, cfg=None, request=None, definition=None):
        return build_performance_plan(
            hw or hardware(), cfg or config(), definition or model(),
            request or params(362, "1920x1088"),
            {}, output_type="video", profile=4,
        )

    def test_auto_owns_split_independently_of_explicit_placement_profile(self):
        cfg = config()
        plan = self.make_plan(
            cfg=cfg,
            request=params(362, "1920x1088", override_profile=4),
        )
        self.assertFalse(plan["applied"])
        self.assertTrue(plan["override_attention_head_split"])
        self.assertEqual(plan["attention_head_split"], 2)
        self.assertTrue(any("explicit profile" in reason for reason in plan["reasons"]))
        self.assertTrue(plan["h3_workload_advice"])
        self.assertTrue(plan["warnings"])

    def test_manual_split_and_auto_off_always_win_but_advice_remains_visible(self):
        manual = config(attention_head_split=1)
        manual_plan = self.make_plan(cfg=manual)
        self.assertFalse(manual_plan["override_attention_head_split"])
        self.assertEqual(manual_plan["attention_head_split"], 1)
        self.assertTrue(manual_plan["h3_workload_advice"])
        self.assertTrue(any("Advice only" in reason for reason in manual_plan["reasons"]))

        auto_off = config()
        auto_off["services"]["auto_performance"] = False
        off_plan = self.make_plan(cfg=auto_off)
        self.assertFalse(off_plan["override_attention_head_split"])
        self.assertEqual(off_plan["attention_head_split"], 0)
        self.assertTrue(off_plan["h3_workload_advice"])

    def test_manual_high_level_is_preserved_and_cuda_false_never_overrides(self):
        high = self.make_plan(cfg=config(attention_head_split=3))
        self.assertEqual(high["attention_head_split"], 3)
        self.assertFalse(high["override_attention_head_split"])
        self.assertTrue(high["h3_workload_advice"])

        no_cuda = self.make_plan(hw=hardware(cuda_available=False))
        self.assertEqual(no_cuda["attention_head_split"], 0)
        self.assertFalse(no_cuda["override_attention_head_split"])

    def test_advice_reasons_warnings_and_workload_fingerprint_include_effective_split(self):
        base = config()
        plan = self.make_plan(cfg=base)
        self.assertTrue(any("estimated target load" in reason for reason in plan["reasons"]))
        self.assertTrue(any("memory guarantee" in warning for warning in plan["warnings"]))
        key_off = workload_key(base, model(), params(362, "1280x704"),
                               effective_attention_head_split=0)
        key_medium = workload_key(base, model(), params(362, "1280x704"),
                                  effective_attention_head_split=2)
        self.assertNotEqual(key_off, key_medium)


if __name__ == "__main__":
    unittest.main()
