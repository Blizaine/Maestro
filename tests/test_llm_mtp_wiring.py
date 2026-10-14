"""Execute the application/Director writer entry points without GPU startup."""

import ast
import asyncio
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from services import llm_service, studio_enhancement


def extract(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(row for row in tree.body if isinstance(row, (ast.FunctionDef, ast.AsyncFunctionDef)) and row.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class MtpEntryPointTests(unittest.TestCase):
    def namespace(self, services):
        return {
            "Request": object, "asyncio": asyncio, "traceback": mock.Mock(),
            "_gen_lock": threading.Lock(), "_DEFAULT_LLM_REPO": "default-writer",
            "_llm_default_device": lambda: "cpu",
            "_guard_interactive_llm_against_generation": lambda: None,
            "enhancement_settings": studio_enhancement.current_settings,
            "wgp": SimpleNamespace(server_config={"services": services}),
            "_wgp": SimpleNamespace(server_config={"services": services}),
        }

    def test_studio_and_director_loads_use_the_captured_mtp_preference(self):
        services = {"llm_model_id": "selected-writer", "llm_device": "cpu", "llm_mtp_acceleration": False}
        entry_points = (
            (ROOT / "app/launch.py", lambda function: function()),
            (ROOT / "app/services/director_pipeline.py", lambda function: function({})),
        )
        for path, call in entry_points:
            function = extract(path, "_ensure_llm_loaded", self.namespace(services))
            for requested in (True, False):
                with self.subTest(path=path.name, requested=requested), \
                     studio_enhancement.enhancement_context({"llm_mtp_acceleration": requested}, lambda: False), \
                     mock.patch.object(llm_service, "load_model") as load:
                    call(function)
                    self.assertIs(load.call_args.kwargs["mtp_acceleration"], requested)
                    self.assertEqual(load.call_args.kwargs["model_id"], "selected-writer")

    def test_interactive_load_uses_saved_setting_without_changing_it(self):
        async def request_json():
            return {}
        request = SimpleNamespace(json=request_json, headers={})
        for requested in (True, False):
            services = {"llm_mtp_acceleration": requested}
            function = extract(ROOT / "app/launch.py", "llm_load", self.namespace(services))
            with self.subTest(requested=requested), \
                 mock.patch.object(llm_service, "load_model") as load, \
                 mock.patch.object(llm_service, "get_status", return_value={"loaded": True}):
                result = asyncio.run(function(request))
                self.assertEqual(result["status"], "ok")
                self.assertIs(load.call_args.kwargs["mtp_acceleration"], requested)
                self.assertEqual(services, {"llm_mtp_acceleration": requested})


if __name__ == "__main__":
    unittest.main()
