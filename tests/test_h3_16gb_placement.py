"""CPU-only coverage of the real H3 guard -> planner -> loader handoff."""
import ast
import contextlib
import copy
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from services.perf_recommend import apply_auto_performance  # noqa: E402
from services import performance_tuning  # noqa: E402


class H3BlackwellPlacementHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((APP / "launch.py").read_text(encoding="utf-8"))
        names = {
            "_stage_count_from_params", "_apply_per_job_coefficient",
            "_restore_base_coefficient", "_apply_per_job_performance",
            "_finish_per_job_performance", "_finalize_deferred_h3_residency_reload",
        }
        constants = {"_BASE_TRANSFORMER_BUDGET_MB", "_H3_RESIDENCY_HEADROOM"}
        nodes = [node for node in tree.body if (
            isinstance(node, ast.FunctionDef) and node.name in names
        ) or (
            isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in constants
                for target in node.targets
            )
        )]
        cls.worker_code = compile(ast.Module(body=nodes, type_ignores=[]), "launch.py", "exec")
        tree = ast.parse((APP / "wgp.py").read_text(encoding="utf-8"))
        node = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "init_pipe")
        cls.loader_code = compile(ast.Module(body=[node], type_ignores=[]), "wgp.py", "exec")
        load = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "load_models")
        stamp = next(node for node in load.body if isinstance(node, ast.Try) and any(
            isinstance(child, ast.Attribute) and child.attr == "_maestro_profile_vram_coefficient"
            for child in ast.walk(node)
        ))
        cls.stamp_code = compile(ast.Module(body=[stamp], type_ignores=[]), "wgp.py", "exec")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.hardware = {
            "cuda_available": True, "gpu_name": "NVIDIA GeForce RTX 5080",
            "gpu_vram_gb": 16.0, "gpu_capability": "12.0", "ram_gb": 63.7,
            "ram_tier": "high", "vram_tier": "low", "platform": "win32",
        }
        self.config = {}
        apply_auto_performance(self.config, self.hardware, force=True)
        self.definition = {
            "architecture": "minimax_h3", "minimax_h3_transformer_working_vram_gb": 10,
        }
        self.runtime = SimpleNamespace(
            args=SimpleNamespace(profile=-1, preload=0, transformer_budget=0,
                                 perc_reserved_mem_max=0, vram_safety_coefficient=.8),
            server_config=self.config, force_profile_no=-1,
            offload=SimpleNamespace(total_pinned_bytes=0), wan_model=None, reload_needed=False,
            get_lora_dir=lambda model: None, get_base_model_type=lambda model: model,
            get_model_def=lambda model: self.definition,
            get_output_type_for_model=lambda *args: "video",
            compute_profile=lambda override, output: (
                self.config["video_profile"] if override == -1 else override
            ),
        )
        history = performance_tuning.PerformanceHistory(Path(self.temp.name) / "history.sqlite3")
        self.context = {
            "wgp": self.runtime, "os": os,
            "_get_cached_hardware": lambda: self.hardware,
            "_performance_hardware": lambda: self.hardware,
            "_get_performance_history": lambda: history,
            "is_cancel_requested": lambda job: False,
        }
        exec(self.worker_code, self.context)
        self.addCleanup(self.context["_restore_base_coefficient"])
        self.loader = {"args": self.runtime.args, "server_config": self.config}
        exec(self.loader_code, self.loader)

    def handoff(self, *, resolution="864x480", frames=124):
        params = {
            "model_type": "minimax_h3", "resolution": resolution,
            "video_length": frames, "sliding_window_size": frames,
            "num_inference_steps": 20, "seed": 12345,
        }
        original_params = copy.deepcopy(params)
        job = {"id": "cpu-placement-test", "params": params, "output_files": []}
        raw = dict(params)
        stats = SimpleNamespace(get_live_stats=lambda: {
            "ram": {"available_gb": 48}, "gpu": {"vram_used_gb": 0},
        })
        monitor = SimpleNamespace(start=lambda: "monitor", finish=lambda: {"samples": 0})
        with contextlib.redirect_stdout(io.StringIO()) as output, patch.dict(
            sys.modules, {"services.live_stats": stats}
        ), patch.object(performance_tuning, "GenerationPerformanceMonitor", return_value=monitor):
            self.context["_apply_per_job_coefficient"](job, defer_h3_residency_reload=True)
            self.assertNotIn("per-job adjustment failed", output.getvalue())
            self.assertEqual(self.context["_apply_per_job_performance"](job, raw), "monitor")
        self.assertEqual(job["params"], original_params)
        options = {"workingVRAM": {"transformer": 10240}, "budgets": {"vae": 512}}
        self.loader["init_pipe"]({"transformer": object()}, options,
                                 self.runtime.compute_profile(raw.get("override_profile", -1), "video"))
        return job, raw, options, monitor

    def test_guarded_allowance_reaches_native_loader_without_changing_saved_profile(self):
        before = copy.deepcopy(self.config)
        job, raw, options, monitor = self.handoff()
        self.assertEqual(raw["override_profile"], 4)
        plan = job["performance_plan"]
        candidate = job["vram_adjustment"]["h3_residency_candidate_mb"]
        self.assertGreater(plan["transformer_budget_mb"], 3072)
        self.assertLessEqual(plan["transformer_budget_mb"], candidate)
        self.assertEqual(options["budgets"]["transformer"], plan["transformer_budget_mb"])
        self.assertEqual(options["workingVRAM"], {"transformer": 10240})
        self.assertEqual(self.config, before)
        self.context["_finish_per_job_performance"](job, monitor, False)
        self.context["_restore_base_coefficient"]()
        self.assertEqual(self.runtime.args.transformer_budget, 0)
        self.assertAlmostEqual(self.runtime.args.vram_safety_coefficient, .8)
        self.assertIsNone(self.runtime.args._maestro_per_job_memory_plan)

    def test_clamped_long_video_retains_profile_default_in_native_loader(self):
        job, raw, options, _ = self.handoff(resolution="1280x704", frames=345)
        self.assertTrue(job["vram_adjustment"]["h3_activation_reserve_clamped"])
        self.assertEqual(raw["override_profile"], 4)
        self.assertEqual(job["performance_plan"]["transformer_budget_mb"], 0)
        self.assertEqual(options["budgets"]["transformer"], 1200)
        self.assertEqual(options["workingVRAM"], {"transformer": 10240})

    def test_cached_old_placement_reloads_once_then_reuses_matching_placement(self):
        self.runtime.wan_model = SimpleNamespace(
            _maestro_performance_placement=(2, 3072, True, .4),
            _maestro_profile_vram_coefficient=.8,
            _maestro_profile_transformer_budget_override_mb=3072,
        )
        job, _, options, monitor = self.handoff()
        self.assertTrue(self.runtime.reload_needed)
        exec(self.stamp_code, {
            "wan_model": self.runtime.wan_model, "kwargs": options,
            "vram_safety_coefficient": self.runtime.args.vram_safety_coefficient,
            "args": self.runtime.args,
        })
        self.context["_finish_per_job_performance"](job, monitor, False)
        self.context["_restore_base_coefficient"]()
        self.runtime.reload_needed = False
        repeated, _, _, repeated_monitor = self.handoff()
        self.assertFalse(self.runtime.reload_needed)
        self.assertEqual(repeated["performance_plan"]["profile"], 4)
        self.context["_finish_per_job_performance"](repeated, repeated_monitor, False)


if __name__ == "__main__":
    unittest.main()
