"""CPU-only regression coverage for H3 guard-to-Auto placement ownership."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services import performance_tuning  # noqa: E402
from services.perf_recommend import apply_auto_performance  # noqa: E402
from services.performance_tuning import build_performance_plan  # noqa: E402


class AutoH3PlacementReloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
        functions = {
            "_stage_count_from_params",
            "_apply_per_job_coefficient",
            "_apply_per_job_performance",
            "_finalize_deferred_h3_residency_reload",
            "_restore_base_coefficient",
        }
        constants = {"_BASE_TRANSFORMER_BUDGET_MB", "_H3_RESIDENCY_HEADROOM"}
        nodes = [
            node
            for node in source.body
            if (
                isinstance(node, ast.FunctionDef)
                and node.name in functions
            )
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in constants
                    for target in node.targets
                )
            )
        ]
        cls.launch_code = compile(ast.Module(body=nodes, type_ignores=[]), "launch.py", "exec")

    def setUp(self):
        self.hardware = {
            "cuda_available": True,
            "gpu_name": "Test GPU",
            "gpu_vram_gb": 12.5,
            "ram_gb": 32,
            "ram_tier": "low",
            "vram_tier": "low",
            "platform": "win32",
        }
        self.config = {}
        apply_auto_performance(self.config, self.hardware, force=True)
        self.history = SimpleNamespace(
            choose=lambda plan, vram, ram, profiles, *, max_budget_mb, clamped=False:
                ({"profile": 4, "budget": 2304}, 2)
                if not clamped and 4 in profiles and max_budget_mb >= 2304
                else (None, 0)
        )
        self.config["vram_safety_coefficient"] = 0.8
        self.assertEqual(self.config["video_profile"], 4)
        self.model_def = {
            "architecture": "minimax_h3",
            "minimax_h3_viggle": True,
            "minimax_h3_full_checkpoint": False,
        }
        self.args = SimpleNamespace(
            preload=0,
            profile="-1",
            transformer_budget=0,
            vram_safety_coefficient=0.8,
            perc_reserved_mem_max=0,
        )
        self.wgp = SimpleNamespace(
            args=self.args,
            server_config=self.config,
            offload=SimpleNamespace(total_pinned_bytes=0),
            force_profile_no=-1,
            wan_model=None,
            reload_needed=False,
            get_output_type_for_model=lambda *args: "video",
            compute_profile=lambda override, output: (
                self.config["video_profile"] if float(override) < 0 else float(override)
            ),
            get_model_def=lambda model: self.model_def,
            get_base_model_type=lambda model: "minimax_h3",
            get_lora_dir=lambda model: None,
        )
        self.live_stats = {
            "ram": {"available_gb": 22},
            "gpu": {"vram_used_gb": 0},
        }
        self.context = {
            "wgp": self.wgp,
            "_get_cached_hardware": lambda: self.hardware,
            "_performance_hardware": lambda: self.hardware,
            "_get_performance_history": lambda: self.history,
            "_BASE_TRANSFORMER_BUDGET_MB": None,
            "_last_performance_plan": None,
            "_H3_RESIDENCY_HEADROOM": 0.97,
            "os": os,
            "json": json,
            "is_cancel_requested": lambda job: False,
            "_workspace_dir": lambda *args: str(ROOT),
        }
        exec(self.launch_code, self.context)
        live_stats = SimpleNamespace(get_live_stats=lambda: self.live_stats)
        self.live_stats_patch = patch.dict(sys.modules, {"services.live_stats": live_stats})
        self.live_stats_patch.start()
        self.addCleanup(self.live_stats_patch.stop)
        self.monitor = SimpleNamespace(start=lambda: "monitor", finish=lambda: {"samples": 0})
        self.monitor_patch = patch.object(
            performance_tuning,
            "GenerationPerformanceMonitor",
            return_value=self.monitor,
        )
        self.monitor_patch.start()
        self.addCleanup(self.monitor_patch.stop)
        self.addCleanup(self.restore_runtime)
        self._install_warm_model()

    def restore_runtime(self):
        self.context["_restore_base_coefficient"]()

    def job(self):
        params = {
            "model_type": "viggle_animate",
            "override_profile": -1,
            "resolution": "864x480",
            "video_length": 124,
            "sliding_window_size": 124,
            "num_inference_steps": 20,
            "activated_loras": [],
        }
        return {"id": "auto-h3-reload-test", "params": params}

    def apply_guard(self, job):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.context["_apply_per_job_coefficient"](
                job, defer_h3_residency_reload=True
            )
        self.assertNotIn("per-job adjustment failed", output.getvalue())
        return output.getvalue()

    def expected_plan(self, adjustment, params):
        cli = {
            "profile": -1,
            "preload": 0,
            "transformer_budget": 0,
            "perc_reserved_mem_max": 0,
        }
        return build_performance_plan(
            self.hardware,
            self.config,
            self.model_def,
            params,
            adjustment,
            output_type="video",
            profile=4,
            cli=cli,
            available_ram_gb=22,
            external_vram_gb=0,
            own_pinned_gb=0,
            history=self.history,
        )

    def _install_warm_model(self):
        # First derive the same guard adjustment without a resident model, then
        # stamp the expected applied placement as an already-loaded H3 profile.
        warmup = self.job()
        self.apply_guard(warmup)
        adjustment = warmup["vram_adjustment"]
        plan = self.expected_plan(adjustment, warmup["params"])
        self.assertTrue(plan["applied"])
        self.assertEqual(plan["profile"], 4)
        self.assertEqual(plan["transformer_budget_mb"], 2304)
        self.context["_finalize_deferred_h3_residency_reload"](
            warmup, plan["transformer_budget_mb"]
        )
        effective_coefficient = adjustment["effective_coef"]
        self.context["_restore_base_coefficient"]()
        self.args.transformer_budget = 0
        self.args.vram_safety_coefficient = 0.8
        self.context["_BASE_TRANSFORMER_BUDGET_MB"] = None
        self.wgp.wan_model = SimpleNamespace(
            _maestro_profile_vram_coefficient=effective_coefficient,
            _maestro_profile_transformer_budget_override_mb=2304,
            _maestro_performance_placement=(
                plan["profile"],
                plan["transformer_budget_mb"],
                plan["read_ahead"],
                plan["reserved_ram_fraction"],
            ),
        )
        self.expected_placement = self.wgp.wan_model._maestro_performance_placement
        self.wgp.reload_needed = False

    def run_worker_plan(self, job, *, budget_override=None, expect_failure=False):
        real_builder = performance_tuning.build_performance_plan

        def builder(*args, **kwargs):
            if expect_failure:
                raise RuntimeError("synthetic planner failure")
            plan = real_builder(*args, **kwargs)
            if budget_override is not None and plan["applied"]:
                plan["transformer_budget_mb"] = budget_override
            return plan

        with patch.object(performance_tuning, "build_performance_plan", side_effect=builder):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.context["_apply_per_job_performance"](job, dict(job["params"]))
        if expect_failure:
            self.assertIn("Per-generation learning unavailable", output.getvalue())
        else:
            self.assertNotIn("Per-generation learning unavailable", output.getvalue())
        return output.getvalue()

    def run_guard_and_worker(self, job, *, budget_override=None):
        guard_log = self.apply_guard(job)
        planner_log = self.run_worker_plan(job, budget_override=budget_override)
        return guard_log + planner_log

    def test_stable_auto_placement_reuses_matching_h3_budget_across_repeats(self):
        for repeat in range(2):
            with self.subTest(repeat=repeat + 1):
                if repeat:
                    self.context["_restore_base_coefficient"]()
                    self.wgp.reload_needed = False
                job = self.job()
                logs = self.run_guard_and_worker(job)
                self.assertEqual(job["performance_plan"]["profile"], 4)
                self.assertEqual(job["performance_plan"]["transformer_budget_mb"], 2304)
                self.assertEqual(self.args.transformer_budget, 2304)
                self.assertEqual(job["vram_adjustment"]["h3_residency_mb"], 2304)
                self.assertGreater(job["vram_adjustment"]["h3_residency_candidate_mb"], 2304)
                self.assertEqual(self.wgp.args._maestro_performance_placement, self.expected_placement)
                self.assertFalse(self.wgp.reload_needed)
                self.assertIn("residency candidate", logs)
                self.assertIn("Profile 4, transformer 2304 MB", logs)
                self.assertNotIn("resident H3 profile will reload", logs)

    def test_changed_final_auto_budget_reloads_and_updates_effective_adjustment(self):
        job = self.job()
        self.run_guard_and_worker(job, budget_override=2048)
        self.assertTrue(self.wgp.reload_needed)
        self.assertEqual(self.args.transformer_budget, 2048)
        self.assertEqual(job["vram_adjustment"]["h3_residency_mb"], 2048)
        self.assertGreater(job["vram_adjustment"]["h3_residency_candidate_mb"], 2048)
        self.assertTrue(
            any("finalized 2048 MB transformer residency budget" in reason
                for reason in job["vram_adjustment"]["reasons"])
        )

    def test_activation_coefficient_mismatch_still_reloads(self):
        self.wgp.wan_model._maestro_profile_vram_coefficient = 1.0
        job = self.job()
        self.run_guard_and_worker(job)
        self.assertTrue(self.wgp.reload_needed)
        self.assertEqual(self.args.transformer_budget, 2304)
        self.assertEqual(self.wgp.args._maestro_performance_placement, self.expected_placement)


    def test_planner_exception_falls_back_to_guard_budget(self):
        job = self.job()
        self.apply_guard(job)
        candidate = job["vram_adjustment"]["h3_residency_candidate_mb"]
        self.assertFalse(self.wgp.reload_needed)
        log = self.run_worker_plan(job, expect_failure=True)
        self.assertIn("Per-generation learning unavailable", log)
        self.assertTrue(self.wgp.reload_needed)
        self.assertEqual(self.args.transformer_budget, candidate)
        self.assertEqual(job["vram_adjustment"]["h3_residency_mb"], candidate)
        self.assertNotIn("_maestro_h3_residency_reload_pending", job)

    def test_slot_owned_path_finalizes_guard_budget_without_auto_planner(self):
        job = self.job()
        self.apply_guard(job)
        candidate = job["vram_adjustment"]["h3_residency_candidate_mb"]
        self.assertFalse(self.wgp.reload_needed)
        self.context["_finalize_deferred_h3_residency_reload"](job)
        self.assertTrue(self.wgp.reload_needed)
        self.assertEqual(self.args.transformer_budget, candidate)
        self.assertEqual(job["vram_adjustment"]["h3_residency_mb"], candidate)
        self.assertNotIn("_maestro_h3_residency_reload_pending", job)

    def test_standalone_guard_keeps_immediate_budget_mismatch_behavior(self):
        job = self.job()
        with contextlib.redirect_stdout(io.StringIO()):
            self.context["_apply_per_job_coefficient"](job)
        self.assertTrue(self.wgp.reload_needed)
        self.assertNotIn("_maestro_h3_residency_reload_pending", job)
        self.assertTrue(
            any("current transformer residency budget" in reason
                for reason in job["vram_adjustment"]["reasons"])
        )

    def test_auto_disabled_explicit_budget_mismatch_still_reloads(self):
        self.config["services"]["auto_performance"] = False
        self.args.transformer_budget = 1536
        job = self.job()
        self.run_guard_and_worker(job)
        self.assertTrue(self.wgp.reload_needed)
        self.assertFalse(job["performance_plan"]["applied"])
        self.assertEqual(job["performance_plan"]["transformer_budget_mb"], 1536)
        self.assertEqual(self.args.transformer_budget, 1536)
        self.assertNotIn("h3_residency_candidate_mb", job["vram_adjustment"])


if __name__ == "__main__":
    unittest.main()
