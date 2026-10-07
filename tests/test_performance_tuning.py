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
          "ram_gb": ram, "ram_tier": "high" if ram >= 64 else "low",
          "vram_tier": "high" if vram >= 24 else "low" if vram >= 12 else "tight",
          "platform": "win32"}
    config = {}
    apply_auto_performance(config, hw, force=True)
    return hw, config


MODEL = {"architecture": "minimax_h3", "minimax_h3_full_checkpoint": False}
PARAMS = {"model_type": "test_h3", "resolution": "1280x704", "video_length": 243,
          "sliding_window_size": 243, "num_inference_steps": 10}
GUARD = {"h3_weight_budget_gb": 12, "effective_coef": .75}


class PlannerTests(unittest.TestCase):
    def plan(self, ram=128, vram=24, **kwargs):
        hw, config = setup(ram, vram)
        return build_performance_plan(hw, config, MODEL, PARAMS, GUARD,
                                      profile=config["video_profile"], **kwargs)

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
        names = {"_apply_per_job_performance", "_finish_per_job_performance"}
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
