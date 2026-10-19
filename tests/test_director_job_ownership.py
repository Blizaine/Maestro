"""Exercise Director child ownership through both public job response paths."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest


LAUNCH = Path(__file__).resolve().parents[1] / "app" / "launch.py"


def load_routes(jobs):
    tree = ast.parse(LAUNCH.read_text(encoding="utf-8"))
    names = {"_director_job_response_fields", "get_status", "list_jobs"}
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(functions) == len(names)
    for node in functions:
        node.decorator_list = []
    namespace = {
        "_jobs": jobs,
        "snapshot_job": deepcopy,
        "public_enhancement": lambda value, **kwargs: value,
        "_generation_previews": SimpleNamespace(
            fields=lambda job: {"preview": job.get("preview")}
        ),
        "_job_eta_response_fields": lambda job: {
            "current_clip": job.get("current_clip"),
            "total_clips": job.get("total_clips"),
        },
    }
    module = ast.fix_missing_locations(
        ast.Module(body=functions, type_ignores=[])
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


if __name__ == "__main__":
    unittest.main()
