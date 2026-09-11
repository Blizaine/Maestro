"""Exercise the real job planner and MMGP configuration without loading a GPU.

launch.py/wgp.py start the application when imported, so isolate their memory
functions via AST. The budget calculations still use services.perf_recommend.
"""
import ast
import contextlib
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))


class H3ResidencyBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ast.parse((APP / "launch.py").read_text(encoding="utf-8"))
        functions = {"_stage_count_from_params", "_apply_per_job_coefficient", "_restore_base_coefficient"}
        constants = {"_BASE_TRANSFORMER_BUDGET_MB", "_H3_RESIDENCY_HEADROOM"}
        nodes = [node for node in source.body if (
            isinstance(node, ast.FunctionDef) and node.name in functions
        ) or (
            isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in constants for target in node.targets
            )
        )]
        cls.planner_code = compile(ast.Module(body=nodes, type_ignores=[]), "launch.py", "exec")
        source = ast.parse((APP / "wgp.py").read_text(encoding="utf-8"))
        init_node = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "init_pipe")
        cls.init_code = compile(ast.Module(body=[init_node], type_ignores=[]), "wgp.py", "exec")
        load_node = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "load_models")
        # Execute the production metadata recording after profiling, so cache
        # reuse tests verify what load_models actually records, not a fake stamp.
        stamp = next(node for node in load_node.body if isinstance(node, ast.Try) and any(
            isinstance(child, ast.Attribute) and child.attr == "_maestro_profile_vram_coefficient"
            for child in ast.walk(node)
        ))
        cls.stamp_code = compile(ast.Module(body=[stamp], type_ignores=[]), "wgp.py", "exec")

    def setUp(self):
        self.hardware = {"gpu_vram_gb": 18.7}
        self.model_def = {
            "architecture": "minimax_h3",
            "minimax_h3_viggle": True,
            "minimax_h3_transformer_working_vram_gb": 10,
        }
        self.profile = 5
        self.wgp = SimpleNamespace(
            args=SimpleNamespace(preload=0, vram_safety_coefficient=0.8),
            server_config={"vram_safety_coefficient": 0.8},
            get_lora_dir=lambda model: None,
            get_base_model_type=lambda model: model,
            get_model_def=lambda model: self.model_def,
            compute_profile=lambda override, output: self.profile if override == -1 else override,
            get_output_type_for_model=lambda model, image_mode: "image" if image_mode else "video",
            wan_model=None,
            reload_needed=False,
        )
        self.namespace = {"wgp": self.wgp, "os": os, "_get_cached_hardware": lambda: self.hardware}
        exec(self.planner_code, self.namespace)
        self.pipe_namespace = {"args": self.wgp.args, "server_config": self.wgp.server_config}
        exec(self.init_code, self.pipe_namespace)
        self.addCleanup(self.restore)

    def plan(self, **overrides):
        params = {
            "model_type": "viggle_animate", "resolution": "864x480",
            "video_length": 1424, "sliding_window_size": 124,
        }
        params.update(overrides)
        job = {"id": "residency-test", "params": params}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.namespace["_apply_per_job_coefficient"](job)
        self.assertNotIn("per-job adjustment failed", output.getvalue())
        return job

    def restore(self):
        self.namespace["_restore_base_coefficient"]()

    def configure(self):
        options = {"workingVRAM": {"transformer": 10240}, "budgets": {"vae": 512}}
        self.pipe_namespace["init_pipe"]({"transformer": object()}, options, self.profile)
        return options

    def record_loaded_model(self):
        self.wgp.wan_model = SimpleNamespace()
        exec(self.stamp_code, {
            "wan_model": self.wgp.wan_model,
            "kwargs": self.configure(),
            "vram_safety_coefficient": self.wgp.args.vram_safety_coefficient,
        })
        self.wgp.reload_needed = False

    def test_a4500_budget_reaches_mmgp_with_workspace_untouched(self):
        adjustment = self.plan()["vram_adjustment"]
        budget = adjustment["h3_residency_mb"]
        self.assertGreater(budget, 9000)  # Was only 100 MB before this fix.
        ceiling = min(adjustment["h3_weight_budget_gb"], adjustment["effective_coef"] * 18.7)
        self.assertLess(budget, ceiling * 1024)
        options = self.configure()
        self.assertEqual(options["budgets"], {"transformer": budget, "text_encoder": 100, "*": 1000, "vae": 512})
        self.assertEqual(options["workingVRAM"], {"transformer": 10240})

    def test_budget_respects_a_tighter_user_coefficient(self):
        self.wgp.server_config["vram_safety_coefficient"] = 0.4
        adjustment = self.plan()["vram_adjustment"]
        self.assertLess(adjustment["effective_coef"] * 18.7, adjustment["h3_weight_budget_gb"])
        self.assertLess(adjustment["h3_residency_mb"], adjustment["effective_coef"] * 18.7 * 1024)

    def test_single_and_multiwindow_jobs_use_the_same_working_set(self):
        multi = self.plan()["vram_adjustment"]["h3_residency_mb"]
        self.restore()
        single = self.plan(video_length=124)["vram_adjustment"]["h3_residency_mb"]
        self.assertEqual(multi, single)

    def test_h3_frames_and_references_also_receive_their_own_budget(self):
        for model, reference_count in (("minimax_h3_fused_turbo", 0), ("minimax_h3_ref2va", 2)):
            with self.subTest(model=model):
                self.model_def.pop("minimax_h3_viggle", None)
                self.model_def["omni_reference"] = bool(reference_count)
                adjustment = self.plan(
                    model_type=model,
                    minimax_h3_references=[{"type": "video"}] * reference_count,
                )["vram_adjustment"]
                self.assertGreater(adjustment["h3_residency_mb"], 100)
                self.assertLess(adjustment["h3_residency_mb"], adjustment["h3_weight_budget_gb"] * 1024)
                self.restore()

    def test_profiles_that_do_not_stream_keep_their_policy(self):
        for profile in (1, 3, 3.5):
            with self.subTest(profile=profile):
                job = self.plan(override_profile=profile)
                self.assertNotIn("h3_residency_mb", job["vram_adjustment"])
                self.assertFalse(hasattr(self.wgp.args, "transformer_budget"))

    def test_manual_preload_is_not_replaced(self):
        for cli, saved in ((5000, 0), (0, 6000), (5000, 6000)):
            with self.subTest(cli=cli, saved=saved):
                self.wgp.args.preload = cli
                self.wgp.server_config["preload_in_VRAM"] = saved
                job = self.plan()
                self.assertNotIn("h3_residency_mb", job["vram_adjustment"])
                self.assertFalse(hasattr(self.wgp.args, "transformer_budget"))
                self.assertEqual(self.configure()["budgets"]["transformer"], cli or saved)

    def test_old_profile_is_reloaded_and_unchanged_jobs_reuse_new_profile(self):
        self.wgp.wan_model = SimpleNamespace(_maestro_profile_vram_coefficient=0.5)
        self.plan()
        self.assertTrue(self.wgp.reload_needed)
        self.record_loaded_model()
        self.restore()
        self.plan()
        self.assertFalse(self.wgp.reload_needed)

    def test_changed_residency_reloads_for_heavier_and_lighter_jobs(self):
        self.plan()
        self.record_loaded_model()
        first = self.wgp.wan_model._maestro_profile_transformer_budget_mb
        self.restore()
        heavier = self.plan(resolution="1280x720", sliding_window_size=240)["vram_adjustment"]
        self.assertLess(heavier["h3_residency_mb"], first)
        self.assertTrue(self.wgp.reload_needed)
        self.record_loaded_model()
        self.restore()
        lighter = self.plan()["vram_adjustment"]
        self.assertEqual(lighter["h3_residency_mb"], first)
        self.assertTrue(self.wgp.reload_needed)

    def test_restore_prevents_budget_leaking_to_another_model(self):
        try:
            self.plan()
            raise RuntimeError("generation cancelled or failed")
        except RuntimeError:
            pass
        finally:
            self.restore()
        self.assertEqual(self.wgp.args.transformer_budget, 0)
        self.assertEqual(self.wgp.args.vram_safety_coefficient, 0.8)
        self.model_def = {"architecture": "wan"}
        self.plan(model_type="wan")
        self.assertEqual(self.configure()["budgets"]["transformer"], 100)

    def test_restore_preserves_prior_value_and_captures_each_jobs_base(self):
        for prior in (512, 1024):
            with self.subTest(prior=prior):
                self.wgp.args.transformer_budget = prior
                self.plan()
                self.restore()
                self.assertEqual(self.wgp.args.transformer_budget, prior)

    def test_budget_restores_even_when_coefficient_restore_fails(self):
        self.plan()
        self.wgp.server_config["vram_safety_coefficient"] = "invalid"
        self.restore()
        self.assertEqual(self.wgp.args.transformer_budget, 0)

    def test_skipped_jobs_do_not_introduce_a_budget(self):
        self.plan(sfx_mode=True)
        self.hardware["gpu_vram_gb"] = 0
        self.plan()
        self.assertFalse(hasattr(self.wgp.args, "transformer_budget"))


if __name__ == "__main__":
    unittest.main()
