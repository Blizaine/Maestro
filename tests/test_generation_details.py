"""Failed attempts keep their own settings through polling, refresh and restart."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services.generation_details import is_terminal_studio_failure, public_generation_details
from services.job_lifecycle import snapshot_job
from services.studio_enhancement import StudioJobArchive


def failed_job(**overrides):
    return {
        "id": "failed1", "status": "failed", "progress": 0, "step": 0,
        "total_steps": 8, "phase": "", "message": "Adapter load failed",
        "error": "Incompatible adapter", "output_files": [], "created_at": 100,
        "workspace": "Test", "params": {
            "model_type": "custom_h3_frames", "prompt": "A boat moves.",
            "num_inference_steps": 8, "seed": 42, "resolution": "704x1280",
            "activated_loras": ["turbo.safetensors"], "loras_multipliers": "0.80",
            "minimax_h3_turbo_mode": False,
        }, **overrides,
    }


class GenerationDetailsTests(unittest.TestCase):
    def test_snapshot_contains_attempt_settings_and_zero_false_values(self):
        job = failed_job()
        job["params"].update(seed=0, guidance_scale=0)
        details = public_generation_details(job)
        self.assertEqual(details["workspace"], "Test")
        self.assertEqual(details["params"]["seed"], 0)
        self.assertEqual(details["params"]["guidance_scale"], 0)
        self.assertIs(details["params"]["minimax_h3_turbo_mode"], False)
        self.assertEqual(details["params"]["activated_loras"], ["turbo.safetensors"])
        self.assertEqual(details["params"]["loras_multipliers"], "0.80")

    def test_public_details_do_not_expose_provider_keys_paths_or_signed_urls(self):
        job = failed_job()
        job["params"].update(
            api_key="private", llm_remote_url="https://private.invalid", image_start="C:/private/input.png",
            custom_settings={"token": "private"}, seed=float("nan"),
            activated_loras=["https://models.invalid/adapter%20one.safetensors?token=private#private",
                             "C:\\private\\adapter-two.safetensors", 123],
        )
        job["performance_plan"] = {"profile": 2, "transformer_budget_mb": 4096,
                                   "private_path": "private", "hardware_key": "private"}
        details = public_generation_details(job)
        self.assertNotIn("private", repr(details))
        self.assertNotIn("seed", details["params"])
        self.assertEqual(details["params"]["activated_loras"], ["adapter one.safetensors", "adapter-two.safetensors"])
        self.assertEqual(details["performance_plan"], {"profile": 2, "transformer_budget_mb": 4096})
        job["params"]["activated_loras"] = "https://models.invalid/one.safetensors?token=private"
        self.assertEqual(public_generation_details(job)["params"]["activated_loras"], ["one.safetensors"])

    def test_missing_settings_are_not_fabricated_from_defaults(self):
        self.assertIsNone(public_generation_details({"id": "legacy"}))
        self.assertEqual(public_generation_details({"params": {}}), {"params": {}})

    def test_failed_and_cancelled_attempts_survive_restart_without_becoming_runnable(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = StudioJobArchive(directory)
            for status in ("failed", "cancelled"):
                original = failed_job(status=status)
                archive.save(original)
                original["params"]["activated_loras"].clear()
                recovered = archive.recover()["failed1"]
                self.assertEqual(recovered["status"], status)
                self.assertEqual(recovered["params"]["activated_loras"], ["turbo.safetensors"])
                self.assertEqual(recovered["error"], "Incompatible adapter")
                self.assertNotIn("enhancement", recovered)

    def test_ordinary_active_successful_and_director_jobs_are_not_new_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = StudioJobArchive(directory)
            for status in ("held", "queued", "running", "completed"):
                archive.save(failed_job(status=status))
            child = failed_job()
            child["params"]["_director_pipeline_id"] = "director"
            archive.save(child)
            archive.save(failed_job(kind="editor_export", params={"project": {"private": "project"}}))
            self.assertEqual(archive.recover(), {})
            self.assertFalse(is_terminal_studio_failure(child))

    def test_ordinary_failure_archive_keeps_display_settings_without_private_execution_data(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = StudioJobArchive(directory)
            job = failed_job(out_dir="C:/private/outputs", hidden_runtime={"token": "private"})
            job["params"].update(api_key="private", image_start="C:/private/input.png",
                                 custom_settings={"token": "private"},
                                 activated_loras=["https://models.invalid/one.safetensors?token=private"])
            archive.save(job)
            saved = json.loads(Path(directory, "failed1.json").read_text(encoding="utf-8"))
            self.assertNotIn("private", repr(saved))
            self.assertEqual(saved["params"]["activated_loras"], ["one.safetensors"])
            self.assertEqual(saved["params"]["num_inference_steps"], 8)
            self.assertEqual(archive.recover()["failed1"]["status"], "failed")

    def test_dismissing_ordinary_failure_removes_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = StudioJobArchive(directory)
            job = failed_job()
            archive.save(job)
            job["dismissed"] = True
            archive.save(job)
            self.assertEqual(archive.recover(), {})

    def test_polling_and_reconnection_return_same_failed_settings(self):
        jobs = {"failed1": failed_job(), "running": failed_job(id="running", status="running", created_at=99),
                "dismissed": failed_job(id="dismissed", dismissed=True),
                "done": failed_job(id="done", status="completed")}
        child = failed_job(id="child")
        child["params"]["_director_pipeline_id"] = "director"
        jobs["child"] = child
        namespace = {"_jobs": jobs, "snapshot_job": snapshot_job,
                     "_gen_lock": SimpleNamespace(locked=lambda: False),
                     "_active_gen_states": {},
                     "_agent_require_job_access": lambda _job_id: None,
                     "principal_context": SimpleNamespace(get=lambda: None),
                     "_is_restricted_agent": lambda _principal: False,
                     "public_generation_details": public_generation_details,
                     "is_terminal_studio_failure": is_terminal_studio_failure,
                     "public_enhancement": lambda record, **_kwargs: record,
                     "_director_job_response_fields": lambda _job: {},
                     "_job_eta_response_fields": lambda _job: {},
                     "_generation_previews": SimpleNamespace(fields=lambda _job: {})}
        tree = ast.parse((ROOT / "app/launch.py").read_text(encoding="utf-8"))
        for name in ("get_status", "list_jobs"):
            function = deepcopy(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name))
            function.decorator_list = []
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "launch.py", "exec"), namespace)
        detail = namespace["get_status"]("failed1")["generation_details"]
        listed = namespace["list_jobs"]()["jobs"]
        self.assertEqual([item["job_id"] for item in listed], ["running", "failed1"])
        self.assertEqual(next(item for item in listed if item["job_id"] == "failed1")["generation_details"], detail)


if __name__ == "__main__":
    unittest.main()
