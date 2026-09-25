"""Exercise the real settings handler without starting the generation engine."""
import ast
import asyncio
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class HTTPError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code


class TestApiKeySaving(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "app/launch.py"
        nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in {"_mask_key", "_persist_server_config", "update_services_config"}]
        for node in nodes:
            node.decorator_list = []
        cls.code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "config.json"
        self.namespace = {
            "Request": object, "HTTPException": HTTPError, "json": json,
            "os": os, "threading": threading,
            "_MODEL_VISIBILITY_WRITE_LOCK": threading.Lock(),
            "_PUBLIC_LLM_PROVIDERS": {"openai", "anthropic"},
            "wgp": SimpleNamespace(server_config={"services": {}},
                                   server_config_filename=str(self.path)),
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

    def test_real_ellipsis_key_is_saved_and_response_is_masked(self):
        key = "abcd...efgh1234567890"
        result = self.save({"civitai_api_key": key})
        self.assertEqual(json.loads(self.path.read_text())["services"]["civitai_api_key"], key)
        self.assertEqual(result["updated"]["civitai_api_key"], "abcd...7890")
        self.assertNotIn(key, json.dumps(result))

    def test_only_the_current_display_mask_is_a_noop(self):
        for key in ("secret", "long-existing-secret"):
            with self.subTest(key=key):
                self.services["openai_api_key"] = key
                mask = self.namespace["_mask_key"](key)
                self.assertEqual(self.save({"openai_api_key": mask})["updated"], {})
                self.assertEqual(self.services["openai_api_key"], key)
        self.save({"openai_api_key": "abcd...wxyz"})
        self.assertEqual(self.services["openai_api_key"], "abcd...wxyz")

    def test_empty_key_clears_and_mixed_mask_request_saves_other_fields(self):
        self.services["openai_api_key"] = "long-existing-secret"
        self.save({"openai_api_key": "long...cret", "llm_device": "cpu"})
        self.assertEqual(self.services["openai_api_key"], "long-existing-secret")
        self.assertEqual(self.services["llm_device"], "cpu")
        self.save({"openai_api_key": ""})
        self.assertEqual(self.services["openai_api_key"], "")

    def test_invalid_values_and_unknown_fields_fail_explicitly(self):
        for payload in ({"openai_api_key": None}, {"openai_api_key": 123}, {"bogus": "ignored"}, []):
            with self.subTest(payload=payload), self.assertRaises(HTTPError) as error:
                self.save(payload)
            self.assertEqual(error.exception.status_code, 400)

    def test_rejected_request_does_not_partially_change_live_settings(self):
        with self.assertRaises(HTTPError):
            self.save({"llm_device": "cpu", "openai_api_key": None})
        self.assertEqual(self.services, {})
        self.assertFalse(self.path.exists())

    def test_failed_save_preserves_previous_key_and_config_file(self):
        self.save({"openai_api_key": "original-secret"})
        previous_file = self.path.read_bytes()
        with patch.object(os, "replace", side_effect=PermissionError("file locked")):
            with self.assertRaises(HTTPError) as error:
                self.save({"openai_api_key": "replacement-secret"})
        self.assertEqual(error.exception.status_code, 500)
        self.assertEqual(self.services["openai_api_key"], "original-secret")
        self.assertEqual(self.path.read_bytes(), previous_file)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_fidelity_preferences_persist_without_changing_other_settings(self):
        self.services["llm_model_id"] = "existing-writer"
        self.save({"enhance_fidelity_retries": 3, "enhance_fidelity_auto_continue": True})
        saved = json.loads(self.path.read_text())["services"]
        self.assertEqual(saved["enhance_fidelity_retries"], 3)
        self.assertTrue(saved["enhance_fidelity_auto_continue"])
        self.assertEqual(saved["llm_model_id"], "existing-writer")
        self.save({"enhance_fidelity_retries": 0, "enhance_fidelity_auto_continue": False})
        self.assertEqual(self.services["enhance_fidelity_retries"], 0)
        self.assertFalse(self.services["enhance_fidelity_auto_continue"])

    def test_fidelity_preferences_reject_invalid_types_and_limits(self):
        for value in (-1, 6, 1.5, True, "3", None):
            with self.subTest(retries=value), self.assertRaises(HTTPError):
                self.save({"enhance_fidelity_retries": value})
        for value in ("false", "true", 0, 1, None):
            with self.subTest(continue_value=value), self.assertRaises(HTTPError):
                self.save({"enhance_fidelity_auto_continue": value})
        self.assertEqual(self.services, {})


if __name__ == "__main__":
    unittest.main()
