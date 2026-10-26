"""CPU-only planner, learning, telemetry and production-worker integration tests."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services.performance_tuning import (  # noqa: E402
    PerformanceHistory, GenerationPerformanceMonitor, build_performance_plan,
    hardware_key, workload_key,
)
from services.perf_recommend import apply_auto_performance  # noqa: E402


def setup(ram=128, vram=24):
    hw = {"cuda_available": True, "gpu_name": "Test GPU", "gpu_vram_gb": vram,
          "ram_gb": ram, "ram_tier": "high" if ram >= 63.5 else "low",
          "vram_tier": "high" if vram >= 24 else "low" if vram >= 12 else "tight",
          "platform": "win32"}
    config = {}
    apply_auto_performance(config, hw, force=True)
    return hw, config


MODEL = {"architecture": "minimax_h3", "minimax_h3_full_checkpoint": False}
PARAMS = {"model_type": "test_h3", "resolution": "1280x704", "video_length": 243,
          "sliding_window_size": 243, "num_inference_steps": 10}
GUARD = {"h3_weight_budget_gb": 12, "effective_coef": .75}


def blackwell_setup(ram=63.7, vram=16, *, capability="sm120", gpu_name="NVIDIA GeForce RTX 5080", platform="win32"):
    hw, config = setup(ram, vram)
    hw.update(gpu_name=gpu_name, platform=platform,
              ram_tier="high" if ram >= 63.5 else "low")
    if capability is None:
        hw.pop("gpu_capability", None)
    else:
        hw["gpu_capability"] = capability
    return hw, config


def guard_with_candidate(weight_budget_gb=9.5):
    return {
        "h3_weight_budget_gb": weight_budget_gb,
        "effective_coef": weight_budget_gb / 16,
        "h3_residency_candidate_mb": int(weight_budget_gb * 1024 * .97),
    }


class PlannerTests(unittest.TestCase):
    def plan(self, ram=128, vram=24, **kwargs):
        hw, config = setup(ram, vram)
        return build_performance_plan(hw, config, MODEL, PARAMS, GUARD,
                                      profile=config["video_profile"], **kwargs)

    def blackwell_plan(self, *, ram=63.7, vram=16, adjustment=None, model=None,
                       params=None, output_type="video", hardware=None, config=None,
                       **kwargs):
        if hardware is None or config is None:
            hardware, config = blackwell_setup(ram, vram)
        plan = build_performance_plan(
            hardware, config, model or MODEL, params or PARAMS, adjustment,
            output_type=output_type,
            profile=config.get(f"{output_type}_profile", config["video_profile"]),
            **kwargs,
        )
        return plan, hardware, config

    def test_ram_allocator_separates_performance_evidence(self):
        hw, _ = setup(32, 12)
        default = hardware_key({**hw, "ram_allocator_active": "default"})
        optimized = hardware_key({**hw, "ram_allocator_active": "mmgp"})
        self.assertNotEqual(default, optimized)

    def test_large_ram_h3_uses_measured_streaming_placement(self):
        plan = self.plan(available_ram_gb=100)
        self.assertTrue(plan["applied"])
        self.assertEqual(plan["profile"], 2)
        self.assertEqual(plan["transformer_budget_mb"], 8192)
        self.assertTrue(plan["read_ahead"])
        self.assertEqual(plan["reserved_ram_fraction"], .4)

    def test_3080_variants_do_not_copy_4090_budget(self):
        for capacity, profile in ((10, 4.5), (12, 4)):
            plan = self.plan(32, capacity, available_ram_gb=22)
            self.assertEqual(plan["profile"], profile)
            self.assertLessEqual(plan["transformer_budget_mb"], 3 * 1024)
            self.assertFalse(plan["read_ahead"])

    def test_model_guard_always_caps_allowance(self):
        hw, config = setup()
        plan = build_performance_plan(hw, config, MODEL, PARAMS,
                                      {"h3_weight_budget_gb": 4, "effective_coef": .8})
        self.assertLessEqual(plan["transformer_budget_mb"], 4 * 1024 * .97)
        clamped = build_performance_plan(hw, config, MODEL, PARAMS,
                                         {**GUARD, "h3_activation_reserve_clamped": True})
        self.assertEqual(clamped["transformer_budget_mb"], 0)
        self.assertTrue(clamped["warnings"])

    def test_blackwell_h3_autotune_preserves_guard_candidate_at_637_and_640_gb_ram(self):
        adjustment = guard_with_candidate(9.5)
        for ram in (63.7, 64.0):
            with self.subTest(ram=ram):
                plan, _, config = self.blackwell_plan(ram=ram, adjustment=adjustment)
                self.assertTrue(plan["applied"])
                self.assertEqual(plan["profile"], 4)
                self.assertEqual(plan["transformer_budget_mb"], adjustment["h3_residency_candidate_mb"])
                self.assertEqual(config["video_profile"], 2)
                self.assertTrue(any("pre-load workspace guard" in reason for reason in plan["reasons"]))
                self.assertTrue(any("selected Profile 4" in reason for reason in plan["reasons"]))

    def test_blackwell_h3_clamped_jobs_keep_profile_default_budget(self):
        adjustment = {
            **guard_with_candidate(3.5),
            "h3_activation_reserve_clamped": True,
            "h3_residency_policy": "profile_default",
            "h3_residency_candidate_mb": 0,
        }
        plan, _, _ = self.blackwell_plan(adjustment=adjustment)
        self.assertEqual(plan["profile"], 4)
        self.assertEqual(plan["transformer_budget_mb"], 0)
        self.assertTrue(plan["warnings"])

    def test_blackwell_h3_candidate_is_capped_by_foreign_vram(self):
        adjustment = guard_with_candidate(9.5)
        plan, _, _ = self.blackwell_plan(adjustment=adjustment, external_vram_gb=4)
        self.assertLess(plan["transformer_budget_mb"], adjustment["h3_residency_candidate_mb"])
        self.assertEqual(plan["transformer_budget_mb"], int((9.5 * .97 - 3.25) * 1024))
        self.assertTrue(plan["warnings"])

    def test_blackwell_h3_missing_or_invalid_candidate_uses_existing_fallback(self):
        guard = guard_with_candidate(9.5)
        missing = dict(guard)
        missing.pop("h3_residency_candidate_mb")
        missing_plan, _, _ = self.blackwell_plan(adjustment=missing)
        self.assertEqual(missing_plan["transformer_budget_mb"], 3072)

        invalid_candidates = (None, 0, -1, 512.5, "nan", float("inf"), True)
        for candidate in invalid_candidates:
            with self.subTest(candidate=candidate):
                adjustment = dict(guard)
                adjustment["h3_residency_candidate_mb"] = candidate
                plan, _, _ = self.blackwell_plan(adjustment=adjustment)
                self.assertEqual(plan["profile"], 4)
                self.assertEqual(plan["transformer_budget_mb"], 3072)

    def test_blackwell_h3_history_cannot_restore_profile2(self):
        class HistoryProbe:
            allowed_profiles = None

            def choose(self, plan, vram, ram, allowed_profiles, *, max_budget_mb, clamped=False):
                self.allowed_profiles = list(allowed_profiles)
                return {"profile": 4.5, "budget": min(4096, max_budget_mb)}, 2

        history = HistoryProbe()
        plan, _, _ = self.blackwell_plan(
            adjustment=guard_with_candidate(9.5), history=history
        )
        self.assertEqual(history.allowed_profiles, [4, 4.5, 5])
        self.assertNotIn(2, history.allowed_profiles)
        self.assertEqual(plan["profile"], 4.5)

    def test_blackwell_h3_host_pressure_can_select_profile5(self):
        plan, _, _ = self.blackwell_plan(
            adjustment=guard_with_candidate(9.5), available_ram_gb=2
        )
        self.assertEqual(plan["profile"], 5)
        self.assertEqual(plan["transformer_budget_mb"], guard_with_candidate(9.5)["h3_residency_candidate_mb"])

    def test_blackwell_h3_requires_standard_video_and_explicit_model_modes(self):
        cases = (
            ({"minimax_h3_fasth3": True}, PARAMS, "video"),
            ({"minimax_h3_fasth3_omniref": True}, PARAMS, "video"),
            ({"minimax_h3_viggle": True}, PARAMS, "video"),
            ({"minimax_h3_audio_only": True}, PARAMS, "video"),
            ({"minimax_h3_image_only": True}, PARAMS, "video"),
            ({"architecture": "minimax_h3_fasth3"}, PARAMS, "video"),
            ({"architecture": "minimax_h3_viggle"}, PARAMS, "video"),
            ({"architecture": "minimax_h3_voice_audio"}, PARAMS, "video"),
            ({"architecture": "minimax_h3_image"}, PARAMS, "video"),
            ({}, {**PARAMS, "audio_only": True}, "video"),
            ({}, {**PARAMS, "image_mode": 1}, "video"),
            ({}, PARAMS, "audio"),
            ({}, PARAMS, "image"),
        )
        for model, params, output_type in cases:
            with self.subTest(model=model, params={key: params.get(key) for key in ("image_mode", "audio_only")}, output_type=output_type):
                plan, _, config = self.blackwell_plan(
                    model={**MODEL, **model}, params=params, output_type=output_type,
                    adjustment=guard_with_candidate(9.5),
                )
                self.assertEqual(plan["profile"], config.get(f"{output_type}_profile", config["video_profile"]))
                self.assertEqual(plan["transformer_budget_mb"], 3072)

    def test_blackwell_h3_policy_requires_target_hardware_and_memory_class(self):
        variants = (
            blackwell_setup(capability="sm89"),
            blackwell_setup(platform="linux"),
            blackwell_setup(ram=32),
            blackwell_setup(ram=63.4),
            blackwell_setup(ram=64.5),
            blackwell_setup(vram=14.9),
            blackwell_setup(vram=16.6),
            blackwell_setup(capability=None, gpu_name="Unknown GPU"),
            blackwell_setup(capability=None, gpu_name="NVIDIA RTX 5000 Ada Generation Laptop GPU"),
        )
        for hardware, config in variants:
            with self.subTest(hardware={key: hardware.get(key) for key in ("gpu_name", "gpu_vram_gb", "gpu_capability", "ram_gb", "platform")}):
                before_profile = config["video_profile"]
                plan, _, _ = self.blackwell_plan(
                    hardware=hardware, config=config, adjustment=guard_with_candidate(9.5)
                )
                generic_hw = {**hardware, "gpu_capability": "sm89"}
                baseline = build_performance_plan(
                    generic_hw, config, MODEL, PARAMS, guard_with_candidate(9.5),
                    profile=config["video_profile"],
                )
                self.assertEqual(plan["profile"], before_profile)
                self.assertEqual(plan["profile"], baseline["profile"])
                self.assertEqual(plan["transformer_budget_mb"], baseline["transformer_budget_mb"])

        # When capability is absent, an explicit RTX 50-series product name is enough.
        hw, config = blackwell_setup(capability=None)
        plan, _, _ = self.blackwell_plan(hardware=hw, config=config, adjustment=guard_with_candidate(9.5))
        self.assertEqual(plan["profile"], 4)
        self.assertEqual(plan["transformer_budget_mb"], guard_with_candidate(9.5)["h3_residency_candidate_mb"])

    def test_blackwell_h3_policy_leaves_8_12_and_24_gb_profiles_unchanged(self):
        for vram in (8, 12, 24):
            with self.subTest(vram=vram):
                hw, config = blackwell_setup(ram=64, vram=vram)
                generic_hw = {**hw, "gpu_name": "Other GPU", "gpu_capability": "sm89"}
                special = build_performance_plan(
                    hw, config, MODEL, PARAMS, GUARD, profile=config["video_profile"]
                )
                baseline = build_performance_plan(
                    generic_hw, config, MODEL, PARAMS, GUARD, profile=config["video_profile"]
                )
                self.assertEqual(special["profile"], baseline["profile"])
                self.assertEqual(special["transformer_budget_mb"], baseline["transformer_budget_mb"])

    def test_blackwell_h3_policy_respects_manual_profile_preload_and_auto_off(self):
        hw, config = blackwell_setup()
        params_before = copy.deepcopy(PARAMS)
        config["video_profile"] = 5
        manual_profile = build_performance_plan(hw, config, MODEL, params_before, GUARD, profile=5)
        self.assertFalse(manual_profile["applied"])
        self.assertEqual(manual_profile["profile"], 5)

        _, config = blackwell_setup()
        config["video_preload_mode"] = "manual"
        manual_preload = build_performance_plan(hw, config, MODEL, params_before, GUARD, profile=config["video_profile"])
        self.assertFalse(manual_preload["applied"])

        _, config = blackwell_setup()
        config["services"]["auto_performance"] = False
        auto_off = build_performance_plan(hw, config, MODEL, params_before, GUARD, profile=config["video_profile"])
        self.assertFalse(auto_off["applied"])
        self.assertEqual(params_before, PARAMS)

    def test_manual_and_cli_choices_preserved(self):
        hw, original = setup()
        for key, value in (("video_profile", 5), ("video_preload_mode", "manual"),
                           ("video_preload_mode", "dynamic")):
            config = copy.deepcopy(original)
            config[key] = value
            self.assertFalse(build_performance_plan(hw, config, MODEL, PARAMS, GUARD)["applied"])
        for cli in ({"profile": 2}, {"preload": 8000}, {"transformer_budget": 1000}):
            self.assertFalse(build_performance_plan(hw, original, MODEL, PARAMS, GUARD, cli=cli)["applied"])
        self.assertFalse(build_performance_plan(hw, original, MODEL, {**PARAMS, "override_profile": 2}, GUARD)["applied"])
        original["services"]["auto_performance"] = False
        self.assertFalse(build_performance_plan(hw, original, MODEL, PARAMS, GUARD)["applied"])

    def test_argparse_string_zero_is_automatic(self):
        plan = self.plan(cli={"profile": "-1", "preload": "0", "transformer_budget": "0", "perc_reserved_mem_max": "0"}, available_ram_gb=100)
        self.assertTrue(plan["applied"])
        self.assertEqual(plan["profile"], 2)
        self.assertTrue(plan["override_reserved_ram"])

    def test_explicit_ram_ceiling_and_read_ahead_not_overridden(self):
        hw, config = setup()
        config.update(perc_reserved_mem_max=20, read_ahead=False)
        plan = build_performance_plan(hw, config, MODEL, PARAMS, GUARD, available_ram_gb=100)
        self.assertEqual(plan["reserved_ram_fraction"], .2)
        self.assertFalse(plan["read_ahead"])
        cli_plan = self.plan(cli={"perc_reserved_mem_max": .1}, available_ram_gb=100)
        self.assertEqual(cli_plan["reserved_ram_fraction"], .1)

    def test_cached_pinned_models_are_reusable_memory(self):
        cold = self.plan(available_ram_gb=100)
        warm = self.plan(available_ram_gb=50, own_pinned_gb=50)
        self.assertEqual(cold["reserved_ram_fraction"], warm["reserved_ram_fraction"])
        self.assertEqual(cold["read_ahead"], warm["read_ahead"])

    def test_foreign_vram_reduces_weight_cap(self):
        plan = self.plan(external_vram_gb=10)
        self.assertLess(plan["transformer_budget_mb"], 8192)
        self.assertTrue(plan["warnings"])

    def test_unknown_existing_memory_settings_have_no_temporary_overrides(self):
        from services.memory_settings import per_job_memory_options
        hw, config = setup()
        config["services"]["auto_performance_defaults"].pop("read_ahead")
        config["services"]["auto_performance_defaults"].pop("perc_reserved_mem_max")
        plan = build_performance_plan(hw, config, {**MODEL, "minimax_h3_full_checkpoint": True}, PARAMS, GUARD, available_ram_gb=5)
        self.assertEqual(plan["reserved_ram_fraction"], .5)
        self.assertEqual(per_job_memory_options(SimpleNamespace(_maestro_per_job_memory_plan=plan)), {})

    def test_pressure_avoids_full_host_pinning(self):
        plan = self.plan(32, 12, available_ram_gb=5)
        self.assertEqual(plan["profile"], 5)
        self.assertLess(plan["reserved_ram_fraction"], .1)
        self.assertTrue(plan["warnings"])

    def test_no_gpu_preserves_config_and_job_quality(self):
        hw, config = setup()
        hw["cuda_available"] = False
        params, before = copy.deepcopy(PARAMS), copy.deepcopy(config)
        plan = build_performance_plan(hw, config, MODEL, params, GUARD)
        self.assertFalse(plan["applied"])
        self.assertEqual(params, PARAMS)
        self.assertEqual(config, before)

    def test_full_h3_ceiling_accounts_for_available_ram(self):
        hw, config = setup()
        model = {**MODEL, "minimax_h3_full_checkpoint": True}
        plan = build_performance_plan(hw, config, model, PARAMS, GUARD, available_ram_gb=100)
        self.assertEqual(plan["reserved_ram_fraction"], .5)
        constrained = build_performance_plan(hw, config, model, PARAMS, GUARD, available_ram_gb=40)
        self.assertLess(constrained["reserved_ram_fraction"], .4)


class LearningTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.history = PerformanceHistory(Path(temp.name) / "history.sqlite3", limit=8)
        self.hw, self.config = setup()
        self.plan = build_performance_plan(self.hw, self.config, MODEL, PARAMS, GUARD, available_ram_gb=100)

    def record(self, ident, budget, seconds, vram=17, completed=True, **changes):
        plan = {**self.plan, "transformer_budget_mb": budget}
        measurement = {"samples": 10, "denoising_seconds": seconds,
                       "peak_vram_gb": vram, "min_available_ram_gb": 50, **changes}
        return self.history.record(ident, plan, measurement, completed=completed)

    def choose(self, **kwargs):
        return self.history.choose(self.plan, 24, 128, [2, 5], max_budget_mb=12000, **kwargs)

    def test_repeated_comparison_prefers_memory_when_timing_tied(self):
        for i in range(2):
            self.record(f"small-{i}", 8192, 101, 17)
            self.record(f"large-{i}", 11000, 100, 21)
        winner, count = self.choose()
        self.assertEqual(count, 4)
        self.assertEqual(winner["budget"], 8192)

    def test_single_samples_and_single_candidate_not_a_winner(self):
        self.record("one", 8192, 100)
        self.record("two", 11000, 101)
        self.assertIsNone(self.choose()[0])
        self.record("three", 8192, 100)
        self.assertIsNone(self.choose()[0])

    def test_unsafe_memory_and_new_workspace_cap_excluded(self):
        for i in range(2):
            self.record(f"safe-{i}", 8192, 100)
            self.record(f"unsafe-{i}", 11000, 90, 23.9)
        self.assertIsNone(self.choose()[0])
        for i in range(2):
            self.record(f"large-{i}", 11000, 90, 20)
        self.assertIsNotNone(self.choose()[0])
        self.assertIsNone(self.history.choose(self.plan, 24, 128, [2], max_budget_mb=9000)[0])
        self.assertIsNone(self.choose(clamped=True)[0])

    def test_cancellation_missing_telemetry_and_duplicates(self):
        self.assertFalse(self.record("cancel", 8192, 100, completed=False))
        self.assertFalse(self.record("missing", 8192, 100, samples=0))
        self.record("dedup", 8192, 100)
        self.record("dedup", 8192, 100)
        self.assertEqual(self.history.summary(self.plan["hardware_key"])["completed_renders"], 1)

    def test_bound_and_reset(self):
        for i in range(12):
            self.record(str(i), 8192, 100)
        self.assertEqual(self.history.summary(self.plan["hardware_key"])["completed_renders"], 8)
        self.history.reset(self.plan["hardware_key"])
        self.assertEqual(self.history.summary(self.plan["hardware_key"])["completed_renders"], 0)

    def test_runtime_checkpoint_references_and_workload_scope(self):
        key = workload_key(self.config, MODEL, PARAMS)
        for params in ({**PARAMS, "video_length": 486}, {**PARAMS, "resolution": "864x480"},
                       {**PARAMS, "image_refs": ["one.png"]}, {**PARAMS, "activated_loras": ["one.safetensors"]}):
            self.assertNotEqual(workload_key(self.config, MODEL, params), key)
        self.assertNotEqual(workload_key(self.config, {**MODEL, "checkpoint_version": 2}, PARAMS), key)
        self.assertNotEqual(hardware_key(self.hw), hardware_key({**self.hw, "driver_version": "new"}))
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "weights"
            file.write_bytes(b"one")
            before = workload_key(self.config, {**MODEL, "path": str(file)}, PARAMS)
            file.write_bytes(b"two2")
            self.assertNotEqual(before, workload_key(self.config, {**MODEL, "path": str(file)}, PARAMS))

    def test_preload_policy_and_amount_scope_each_output_workload(self):
        baseline = workload_key(self.config, MODEL, PARAMS)
        self.assertEqual(baseline, workload_key(copy.deepcopy(self.config), MODEL, PARAMS))

        for kind in ("video", "image", "audio"):
            for mode, amount in (("dynamic", 0), ("manual", 6000)):
                config = copy.deepcopy(self.config)
                config[f"{kind}_preload_mode"] = mode
                config[f"{kind}_preload_in_VRAM"] = amount
                with self.subTest(kind=kind, mode=mode):
                    self.assertNotEqual(baseline, workload_key(config, MODEL, PARAMS))

            manual = copy.deepcopy(self.config)
            manual[f"{kind}_preload_mode"] = "manual"
            manual[f"{kind}_preload_in_VRAM"] = 6000
            changed_amount = copy.deepcopy(manual)
            changed_amount[f"{kind}_preload_in_VRAM"] = 8000
            with self.subTest(kind=kind, changed_amount=True):
                self.assertNotEqual(
                    workload_key(manual, MODEL, PARAMS),
                    workload_key(changed_amount, MODEL, PARAMS),
                )

        legacy_manual = copy.deepcopy(self.config)
        for kind in ("video", "image", "audio"):
            legacy_manual.pop(f"{kind}_preload_mode")
            legacy_manual.pop(f"{kind}_preload_in_VRAM")
        legacy_manual["preload_in_VRAM"] = 6000
        self.assertNotEqual(baseline, workload_key(legacy_manual, MODEL, PARAMS))
        self.assertNotEqual(
            baseline,
            workload_key(self.config, MODEL, PARAMS, cli={"preload": 6000}),
        )

    def test_manual_dynamic_history_cannot_seed_default_static_placement(self):
        dynamic_config = copy.deepcopy(self.config)
        dynamic_config["video_preload_mode"] = "dynamic"
        dynamic_plan = {
            **self.plan,
            "workload_key": workload_key(dynamic_config, MODEL, PARAMS),
        }
        for profile, budget, seconds in ((2, 8192, 101), (5, 11000, 100)):
            for repeat in range(2):
                self.history.record(
                    f"manual-dynamic-{profile}-{repeat}",
                    {**dynamic_plan, "profile": profile, "transformer_budget_mb": budget},
                    {"samples": 10, "denoising_seconds": seconds,
                     "peak_vram_gb": 17, "min_available_ram_gb": 50},
                    completed=True,
                )

        self.assertIsNone(self.choose()[0])
        winner, count = self.history.choose(
            dynamic_plan, 24, 128, [2, 5], max_budget_mb=12000,
        )
        self.assertEqual(count, 4)
        self.assertIsNotNone(winner)

    def test_profile_comparisons_share_preload_fingerprint(self):
        baseline = workload_key(self.config, MODEL, PARAMS)
        other_profile = copy.deepcopy(self.config)
        other_profile["video_profile"] = 5
        self.assertEqual(baseline, workload_key(other_profile, MODEL, PARAMS))

    def test_auto_preload_runtime_identity_scopes_hardware_history(self):
        original = hardware_key(self.hw)
        self.assertEqual(original, hardware_key(copy.deepcopy(self.hw)))
        with patch("services.performance_tuning._auto_preload_runtime_identity", return_value="new-code"):
            self.assertNotEqual(original, hardware_key(self.hw))

    def test_3080_matrix_can_teach_both_vram_variants(self):
        for capacity in (10, 12):
            hw, config = setup(32, capacity)
            guard = {**GUARD, "h3_activation_reserve_clamped": True}
            baseline = build_performance_plan(hw, config, MODEL, PARAMS, guard,
                                              profile=config["video_profile"], available_ram_gb=22)
            for profile, seconds in ((4.5, 50), (5, 60)):
                for repeat in range(2):
                    self.history.record(f"{capacity}-{profile}-{repeat}", {**baseline, "profile": profile},
                                        {"samples": 40, "denoising_seconds": seconds,
                                         "peak_vram_gb": capacity - 1, "min_available_ram_gb": 8}, completed=True)
            learned = build_performance_plan(hw, config, MODEL, PARAMS, guard,
                                            profile=config["video_profile"], available_ram_gb=22, history=self.history)
            self.assertEqual(learned["source"], "local_history")
            self.assertEqual(learned["profile"], 4.5)


class MonitorTests(unittest.TestCase):
    def test_time_phases_and_physical_memory(self):
        clock = [0]
        phase = ["Loading model"]
        stats = {"gpu": {"available": True, "vram_used_gb": 17},
                 "ram": {"available_gb": 50, "total_gb": 128}}
        monitor = GenerationPerformanceMonitor(lambda: stats, lambda: phase[0], clock=lambda: clock[0])
        monitor.sample()
        clock[0], phase[0] = 10, "Denoising"
        monitor.sample()
        clock[0], phase[0] = 30, "Decoding"
        stats["gpu"]["vram_used_gb"] = 21
        monitor.sample()
        clock[0] = 40
        result = monitor.finish()
        self.assertEqual(result["denoising_seconds"], 20)
        self.assertEqual(result["peak_vram_gb"], 21)
        self.assertEqual(result["elapsed_seconds"], 40)

    def test_bad_telemetry_does_not_fail_generation(self):
        def stats():
            raise RuntimeError("NVML unavailable")
        monitor = GenerationPerformanceMonitor(stats, lambda: "Denoising")
        result = monitor.start().finish()
        self.assertEqual(result["samples"], 0)
        self.assertIsNone(result["min_available_ram_gb"])
        self.assertFalse(monitor.thread.is_alive())


class RuntimeIntegrationTests(unittest.TestCase):
    def test_real_worker_sets_profile_budget_and_restores_temporary_options(self):
        from services import performance_tuning
        tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
        names = {"_apply_per_job_performance", "_finish_per_job_performance", "_finalize_deferred_h3_residency_reload"}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        hw, config = setup()
        args = SimpleNamespace(profile="-1", transformer_budget=12000, preload="0", perc_reserved_mem_max=0)
        runtime = SimpleNamespace(server_config=config, args=args, offload=SimpleNamespace(total_pinned_bytes=0),
                                  wan_model=SimpleNamespace(), force_profile_no=-1,
                                  get_output_type_for_model=lambda *a: "video", compute_profile=lambda *a: 1,
                                  get_model_def=lambda *a: MODEL)
        monitor = SimpleNamespace(start=lambda: "monitor", finish=lambda: {"samples": 0})
        with tempfile.TemporaryDirectory() as temp:
            history = PerformanceHistory(Path(temp) / "history.sqlite3")
            context = {"wgp": runtime, "_BASE_TRANSFORMER_BUDGET_MB": 0,
                       "_performance_hardware": lambda: hw, "_get_performance_history": lambda: history,
                       "is_cancel_requested": lambda job: False, "_workspace_dir": lambda *a: temp,
                       "os": __import__("os"), "json": json}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "launch.py", "exec"), context)
            stats = SimpleNamespace(get_live_stats=lambda: {"ram": {"available_gb": 100}})
            with patch.dict(sys.modules, {"services.live_stats": stats}), patch.object(performance_tuning, "GenerationPerformanceMonitor", return_value=monitor):
                job = {"id": "test", "params": PARAMS, "vram_adjustment": GUARD,
                       "out_dir": temp, "output_files": ["render.mp4"]}
                metadata_path = Path(temp) / "render.meta.json"
                metadata_path.write_text(json.dumps({"params": {"seed": 123}}))
                raw = dict(PARAMS)
                self.assertEqual(context["_apply_per_job_performance"](job, raw), "monitor")
                self.assertEqual(raw["override_profile"], 2)
                self.assertEqual(args.transformer_budget, 8192)
                self.assertTrue(runtime.reload_needed)
                self.assertEqual(config["video_profile"], 1)
                context["_finish_per_job_performance"](job, monitor, False)
                self.assertIsNone(args._maestro_per_job_memory_plan)
                self.assertIsNone(args._maestro_performance_placement)
                metadata = json.loads(metadata_path.read_text())
                self.assertEqual(metadata["params"]["seed"], 123)
                self.assertEqual(metadata["performance_plan"]["profile"], 2)
                self.assertEqual(metadata["performance_measurements"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
