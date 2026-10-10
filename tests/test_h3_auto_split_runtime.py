"""CPU-only coverage of the real generation worker's temporary head split."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services.memory_settings import per_job_attention_head_split
from services.perf_recommend import apply_auto_performance
from services import performance_tuning


class HeadSplitBoundaryTests(unittest.TestCase):
    def test_only_a_valid_video_override_changes_saved_grouping(self):
        config = {"attention_head_split": 3}
        for plan in (None, {}, {"output_type": "image", "override_attention_head_split": True, "attention_head_split": 2},
                     {"output_type": "video", "override_attention_head_split": False, "attention_head_split": 2},
                     *({"output_type": "video", "override_attention_head_split": True, "attention_head_split": value}
                       for value in (True, -1, 4, "2", None))):
            with self.subTest(plan=plan):
                self.assertEqual(per_job_attention_head_split(config, SimpleNamespace(_maestro_per_job_memory_plan=plan)), 3)
        args = SimpleNamespace(_maestro_per_job_memory_plan={
            "output_type": "video", "override_attention_head_split": True, "attention_head_split": 0})
        self.assertEqual(per_job_attention_head_split(config, args), 0)
        self.assertEqual(config, {"attention_head_split": 3})

    def test_engine_boundary_consumes_the_temporary_choice(self):
        tree = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))
        boundary = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                        and isinstance(node.value.func.value, ast.Name)
                        and node.value.func.value.id == "attention_kit"
                        and node.value.func.attr == "configure"
                        and any(isinstance(child, ast.Name) and child.id == "per_job_attention_head_split"
                                for child in ast.walk(node)))
        levels = []
        args = SimpleNamespace(_maestro_per_job_memory_plan={
            "output_type": "video", "override_attention_head_split": True, "attention_head_split": 2})
        context = {"attention_kit": SimpleNamespace(configure=levels.append), "server_config": {"attention_head_split": 0},
                   "args": args, "per_job_attention_head_split": per_job_attention_head_split}
        code = compile(ast.Module(body=[boundary], type_ignores=[]), "wgp.py", "exec")
        exec(code, context)
        args._maestro_per_job_memory_plan = None
        exec(code, context)
        self.assertEqual(levels, [2, 0])


class WorkerHeadSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
        names = {"_apply_per_job_performance", "_finish_per_job_performance", "_finalize_deferred_h3_residency_reload"}
        cls.code = compile(ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                                           and node.name in names], type_ignores=[]), "launch.py", "exec")

    def setUp(self):
        self.hw = {"cuda_available": True, "gpu_name": "NVIDIA GeForce RTX 3080 Ti", "gpu_vram_gb": 12,
                   "ram_gb": 31.76, "platform": "win32", "supports_mmgp_allocator": True}
        self.config = {}
        apply_auto_performance(self.config, self.hw, force=True)
        self.before = copy.deepcopy(self.config)
        self.args = SimpleNamespace(profile=-1, transformer_budget=0, preload=0, perc_reserved_mem_max=0)
        model = {"architecture": "minimax_h3_ref2va", "minimax_h3_fused_turbo": True,
                 "minimax_h3_full_checkpoint": False, "sla_attention_default": True}
        self.runtime = SimpleNamespace(server_config=self.config, args=self.args,
            offload=SimpleNamespace(total_pinned_bytes=0), wan_model=None, force_profile_no=-1,
            get_output_type_for_model=lambda *a: "video", compute_profile=lambda *a: 4,
            get_model_def=lambda *a: model, reload_needed=False)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.context = {"wgp": self.runtime, "_BASE_TRANSFORMER_BUDGET_MB": 0,
            "_performance_hardware": lambda: self.hw,
            "_get_performance_history": lambda: SimpleNamespace(choose=lambda *a, **kw: (None, 0), record=lambda *a, **kw: False),
            "is_cancel_requested": lambda job: job.get("status") == "cancelled",
            "_workspace_dir": lambda *a: self.temp.name, "os": os, "json": json}
        exec(self.code, self.context)
        stats = SimpleNamespace(get_live_stats=lambda: {"ram": {"available_gb": 24}, "gpu": {"vram_used_gb": 1}})
        monitor = SimpleNamespace(start=lambda: self.monitor, finish=lambda: {"samples": 0})
        self.monitor = monitor
        patches = [patch.dict(sys.modules, {"services.live_stats": stats}),
                   patch.object(performance_tuning, "GenerationPerformanceMonitor", return_value=monitor)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def apply(self, resolution="1920x1088", frames=362):
        params = {"model_type": "minimax_h3_ref2va_fused_turbo", "resolution": resolution,
                  "video_length": frames, "sliding_window_size": frames, "num_inference_steps": 6,
                  "override_attention": "sla", "override_profile": 4, "activated_loras": []}
        job = {"id": "runtime-test", "params": params, "output_files": []}
        raw = dict(params)
        self.context["_apply_per_job_performance"](job, raw)
        self.assertEqual(raw, params, "Head splitting must not alter generation content or explicit placement")
        return job

    def test_explicit_profile_allows_auto_heads_and_smaller_next_job_resets(self):
        long_job = self.apply()
        self.assertFalse(long_job["performance_plan"]["applied"])
        self.assertTrue(long_job["performance_plan"]["override_attention_head_split"])
        self.assertEqual(per_job_attention_head_split(self.config, self.args), 2)
        signature = self.args._maestro_performance_placement
        self.context["_finish_per_job_performance"](long_job, self.monitor, True)
        self.runtime.wan_model = SimpleNamespace(_maestro_performance_placement=signature)
        short_job = self.apply("1280x704", 124)
        self.assertEqual(per_job_attention_head_split(self.config, self.args), 0)
        self.assertFalse(self.runtime.reload_needed, "Grouping alone must not force checkpoint reload")
        self.context["_finish_per_job_performance"](short_job, self.monitor, True)
        self.assertEqual(self.config, self.before)

    def test_failure_and_cancellation_clear_temporary_override(self):
        for status in ("failed", "cancelled"):
            with self.subTest(status=status):
                job = self.apply()
                self.assertEqual(per_job_attention_head_split(self.config, self.args), 2)
                job["status"] = status
                self.context["_finish_per_job_performance"](job, self.monitor, False)
                self.assertIsNone(self.args._maestro_per_job_memory_plan)
                self.assertEqual(per_job_attention_head_split(self.config, self.args), 0)
        self.assertEqual(self.config, self.before)

    def test_manual_choice_is_retained(self):
        self.config["services"]["auto_performance"] = False
        self.config["attention_head_split"] = 1
        job = self.apply()
        self.assertFalse(job["performance_plan"]["override_attention_head_split"])
        self.assertEqual(per_job_attention_head_split(self.config, self.args), 1)
        self.context["_finish_per_job_performance"](job, self.monitor, True)
        self.assertEqual(self.config["attention_head_split"], 1)


if __name__ == "__main__":
    unittest.main()
