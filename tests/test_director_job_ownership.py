"""Exercise Director child ownership through both public job response paths."""
import ast
import atexit
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from fastapi import HTTPException

import sys

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
from services.agent_access.auth import Principal
from services.agent_access.backend import AgentAccessBackend, is_restricted_agent
from services.agent_access.context import principal_context
from services.generation_details import is_terminal_studio_failure, public_generation_details
from services.job_lifecycle import snapshot_job


LAUNCH = Path(__file__).resolve().parents[1] / "app" / "launch.py"


def load_routes(jobs):
    tree = ast.parse(LAUNCH.read_text(encoding="utf-8"))
    names = {
        "_agent_filter_director_queue", "_agent_require_job_access",
        "_agent_require_queue_collection", "_director_job_response_fields",
        "get_status", "list_jobs",
    }
    constant_names = {"_AGENT_QUEUE_PENDING_STATUSES", "_AGENT_QUEUE_TERMINAL_STATUSES"}
    selected = []
    found_functions = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            selected.append(node)
            found_functions.add(node.name)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constant_names
            for target in node.targets
        ):
            selected.append(node)
    assert found_functions == names
    scratch = tempfile.TemporaryDirectory()
    atexit.register(scratch.cleanup)
    root = Path(scratch.name)
    app_root = root / "app"
    app_root.mkdir()
    runtime = SimpleNamespace(directory=root / "settings", backend=lambda: {})
    backend = AgentAccessBackend(runtime, app_root, recover_previous=False)
    namespace = {
        "_jobs": jobs,
        "_gen_lock": threading.Lock(),
        "_active_gen_states": {},
        "_agent_access_backend": backend,
        "_is_restricted_agent": is_restricted_agent,
        "principal_context": principal_context,
        "HTTPException": HTTPException,
        "snapshot_job": snapshot_job,
        "public_generation_details": public_generation_details,
        "is_terminal_studio_failure": is_terminal_studio_failure,
        "public_enhancement": lambda value, **kwargs: value,
        "_generation_previews": SimpleNamespace(
            fields=lambda job: {"preview": job.get("preview")}
        ),
        "_job_eta_response_fields": lambda job: {
            "current_clip": job.get("current_clip"),
            "total_clips": job.get("total_clips"),
        },
        "_test_scratch": scratch,
    }
    module = ast.fix_missing_locations(
        ast.Module(body=selected, type_ignores=[])
    )
    exec(compile(module, str(LAUNCH), "exec"), namespace)
    return namespace


def job(identifier, params=None, **fields):
    return {
        "id": identifier, "status": "running", "progress": 17,
        "message": "Denoising", "output_files": [], "error": None,
        "params": {} if params is None else params, **fields,
    }


class DirectorJobOwnershipTests(unittest.TestCase):
    def test_both_routes_keep_pipeline_ownership_and_previews_distinct(self):
        jobs = {
            "child-a": job("child-a", {"_director_pipeline_id": "pipeline-a"},
                           preview={"url": "/preview/a", "revision": 2}),
            "child-b": job("child-b", {"_director_pipeline_id": "pipeline-b"},
                           preview={"url": "/preview/b", "revision": 7}),
            "studio": job("studio", {"model_type": "h3"}),
        }
        routes = load_routes(jobs)
        listed = {item["job_id"]: item for item in routes["list_jobs"]()["jobs"]}
        for identifier, owner in (
            ("child-a", "pipeline-a"), ("child-b", "pipeline-b"),
            ("studio", None),
        ):
            with self.subTest(job=identifier):
                status = routes["get_status"](identifier)
                self.assertEqual(status["director_pipeline_id"], owner)
                self.assertEqual(listed[identifier]["director_pipeline_id"], owner)
                self.assertFalse(status["director_detached_operation"])
                self.assertFalse(listed[identifier]["director_detached_operation"])
                self.assertEqual(status["preview"], jobs[identifier].get("preview"))
                self.assertEqual(listed[identifier]["preview"], status["preview"])

    def test_detached_repairs_remain_distinguishable_from_parent_render(self):
        jobs = {
            "main": job("main", {"_director_pipeline_id": "pipeline"}),
            "repair": job("repair", {
                "_director_pipeline_id": "pipeline",
                "_director_detached_operation": True,
            }),
        }
        routes = load_routes(jobs)
        listed = {item["job_id"]: item for item in routes["list_jobs"]()["jobs"]}
        for identifier, detached in (("main", False), ("repair", True)):
            self.assertEqual(
                routes["get_status"](identifier)["director_detached_operation"],
                detached,
            )
            self.assertEqual(listed[identifier]["director_detached_operation"], detached)

    def test_private_generation_params_are_not_published_or_modified(self):
        jobs = {"child": job("child", {
            "_director_pipeline_id": "pipeline",
            "model_path": "private/model/path",
            "api_key": "dummy-secret-for-regression",
        })}
        original = deepcopy(jobs)
        routes = load_routes(jobs)
        responses = [
            routes["get_status"]("child"), routes["list_jobs"]()["jobs"][0]
        ]
        for response in responses:
            self.assertNotIn("params", response)
            self.assertNotIn("private/model/path", str(response))
            self.assertNotIn("dummy-secret-for-regression", str(response))
        self.assertEqual(jobs, original)

    def test_invalid_or_missing_owner_cannot_claim_a_director_pipeline(self):
        for params in (
            None, "invalid", {"_director_pipeline_id": ""},
            {"_director_pipeline_id": "   "},
            {"_director_pipeline_id": {"id": "pipeline"}},
            {"_director_pipeline_id": 123},
            {"_director_detached_operation": True},
        ):
            with self.subTest(params=params):
                jobs = {"job": job("job", params)}
                routes = load_routes(jobs)
                for response in (
                    routes["get_status"]("job"), routes["list_jobs"]()["jobs"][0]
                ):
                    self.assertIsNone(response["director_pipeline_id"])
                    self.assertFalse(response["director_detached_operation"])

    def test_recovery_lists_queued_children_but_not_dismissed_or_finished_jobs(self):
        jobs = {
            "queued": job("queued", {"_director_pipeline_id": "pipeline"},
                          status="queued"),
            "dismissed": job("dismissed", {"_director_pipeline_id": "pipeline"},
                             dismissed=True),
            "finished": job("finished", {"_director_pipeline_id": "pipeline"},
                           status="completed"),
        }
        routes = load_routes(jobs)
        listed = routes["list_jobs"]()["jobs"]
        self.assertEqual([item["job_id"] for item in listed], ["queued"])
        self.assertEqual(listed[0]["director_pipeline_id"], "pipeline")

    def test_authenticated_job_routes_use_real_principal_scoped_backend_helpers(self):
        routes = load_routes({})
        backend = routes["_agent_access_backend"]
        jobs = backend.registry()
        backend.jobs = jobs
        routes["_jobs"] = jobs
        actor_a = Principal("actor-a", "client-a", ("observe", "generate"))
        actor_b = Principal("actor-b", "client-b", ("observe", "generate"))

        token = principal_context.set(actor_a)
        try:
            jobs["owned-a"] = job("owned-a", {"prompt": "private actor A prompt"})
        finally:
            principal_context.reset(token)
        token = principal_context.set(actor_b)
        try:
            jobs["owned-b"] = job("owned-b", {"prompt": "private actor B prompt"})
        finally:
            principal_context.reset(token)

        token = principal_context.set(actor_a)
        try:
            self.assertEqual(routes["get_status"]("owned-a")["job_id"], "owned-a")
            with self.assertRaises(HTTPException) as denied:
                routes["get_status"]("owned-b")
            self.assertEqual(denied.exception.status_code, 404)
            result = routes["list_jobs"]()
        finally:
            principal_context.reset(token)
        self.assertEqual([item["job_id"] for item in result["jobs"]], ["owned-a"])
        self.assertTrue(result["busy"])
        self.assertNotIn("private actor B prompt", str(result))

    def test_director_queue_helpers_filter_other_owners_and_block_global_mutations(self):
        routes = load_routes({})
        backend = routes["_agent_access_backend"]
        actor_a = Principal("actor-a", "client-a", ("observe", "generate"))
        backend.journal.register_resource_owner("queue", "queue-a", actor_a.id, backend.session_id)
        backend.journal.register_resource_owner("queue", "queue-b", "actor-b", backend.session_id)
        queue_state = {
            "running": True,
            "entries": [
                {"id": "queue-a", "status": "held", "scene_description": "my project"},
                {"id": "queue-b", "status": "running", "scene_description": "other project prompt"},
            ],
        }

        token = principal_context.set(actor_a)
        try:
            routes["_agent_require_queue_collection"](queue_state, {"held"})
            with self.assertRaises(HTTPException) as denied:
                routes["_agent_require_queue_collection"](queue_state)
            self.assertEqual(denied.exception.status_code, 403)
            filtered = routes["_agent_filter_director_queue"](queue_state)
        finally:
            principal_context.reset(token)

        self.assertEqual([entry["id"] for entry in filtered["entries"]], ["queue-a"])
        self.assertTrue(filtered["other_work_pending"])
        self.assertTrue(filtered["running"])
        self.assertNotIn("other project prompt", str(filtered))


if __name__ == "__main__":
    unittest.main()
