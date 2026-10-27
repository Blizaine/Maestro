"""Exercise the Worlds visibility preference handler without the model runtime."""
import ast
import asyncio
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest


class HTTPError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code


class TestWorldsSetting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "app/launch.py"
        names = {"_mask_key", "_persist_server_config", "get_services_config", "update_services_config"}
        nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in names]
        for node in nodes:
            node.decorator_list = []
        cls.code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "config.json"
        self.namespace = {
            "Request": object,
            "HTTPException": HTTPError,
            "json": json,
            "os": os,
            "threading": threading,
            "_MODEL_VISIBILITY_WRITE_LOCK": threading.Lock(),
            "_PUBLIC_LLM_PROVIDERS": {"openai", "anthropic"},
            "_DEFAULT_LLM_REPO": "default-writer",
            "_llm_default_device": lambda: "cpu",
            "wgp": SimpleNamespace(
                server_config={"services": {}},
                server_config_filename=str(self.path),
            ),
        }
        exec(self.code, self.namespace)

    @property
    def services(self):
        return self.namespace["wgp"].server_config["services"]

    def save(self, payload):
        async def request_json():
            return payload

        return asyncio.run(self.namespace["update_services_config"](
            SimpleNamespace(json=request_json)))

    def test_missing_preference_defaults_off_and_round_trips(self):
        self.assertIs(self.namespace["get_services_config"]()["worlds_enabled"], False)
        self.services["llm_model_id"] = "existing-writer"

        self.save({"worlds_enabled": True})
        self.assertIs(self.namespace["get_services_config"]()["worlds_enabled"], True)
        saved = json.loads(self.path.read_text(encoding="utf-8"))["services"]
        self.assertIs(saved["worlds_enabled"], True)
        self.assertEqual(saved["llm_model_id"], "existing-writer")

        self.save({"worlds_enabled": False})
        self.assertIs(self.namespace["get_services_config"]()["worlds_enabled"], False)
        saved = json.loads(self.path.read_text(encoding="utf-8"))["services"]
        self.assertIs(saved["worlds_enabled"], False)
        self.assertEqual(saved["llm_model_id"], "existing-writer")

    def test_invalid_values_are_rejected_before_any_write_or_mutation(self):
        self.services["llm_model_id"] = "existing-writer"
        self.save({"llm_device": "cpu"})
        previous_services = dict(self.services)
        previous_file = self.path.read_bytes()
        persist_calls = []
        original_persist = self.namespace["_persist_server_config"]
        self.namespace["_persist_server_config"] = lambda: persist_calls.append(True)
        self.addCleanup(lambda: self.namespace.__setitem__("_persist_server_config", original_persist))

        for value in ("false", "true", 0, 1, None, [], {}):
            with self.subTest(value=value), self.assertRaises(HTTPError) as error:
                self.save({"llm_device": "cuda", "worlds_enabled": value})
            self.assertEqual(error.exception.status_code, 400)
            self.assertEqual(self.services, previous_services)
            self.assertEqual(self.path.read_bytes(), previous_file)

        self.assertEqual(persist_calls, [])


if __name__ == "__main__":
    unittest.main()
