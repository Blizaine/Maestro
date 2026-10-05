"""Incomplete speech must fail before H3 planning or generation starts."""
from pathlib import Path
import ast
import asyncio
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
INCOMPLETE = (
    'Alex says, "Welcome." Sam replies, "Look at me!" '
    'Casey says in a frustrated voice "Thanks a lot. You son of a '
    '[beep censor]. The audience laughs.'
)
COMPLETE = (
    'Alex says, "Welcome." Sam replies, "Look at me!" '
    'Casey says in a frustrated voice "Thanks a lot." The audience laughs.'
)


class HTTPError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


def load_routes():
    # Execute the real early route logic without importing the running app,
    # its generation workers, model loaders, or FastAPI service lifecycle.
    path = APP / "launch.py"
    names = {
        "_is_minimax_h3_identity", "_validate_h3_source_quote_boundaries",
        "llm_plan_h3_windows", "llm_plan_h3_sequence",
        "_prepare_generation_submission",
    }
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    namespace = {
        "HTTPException": HTTPError,
        "_stamp_live_generation_settings_version": Mock(),
        "wgp": SimpleNamespace(
            get_model_def=Mock(return_value={"architecture": "minimax_h3_ref2va"}),
            get_base_model_type=Mock(return_value="minimax_h3_ref2va"),
        ),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


class QuoteAPIAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routes = load_routes()

    def setUp(self):
        self.routes["wgp"].get_model_def.reset_mock()
        self.routes["wgp"].get_model_def.side_effect = None

    def test_incomplete_speech_stops_each_window_endpoint_before_model_access(self):
        for name in ("llm_plan_h3_windows", "llm_plan_h3_sequence"):
            with self.subTest(endpoint=name):
                with self.assertRaises(HTTPError) as raised:
                    asyncio.run(self.routes[name](Request({"prompt": INCOMPLETE})))
                self.assertEqual(raised.exception.status_code, 400)
                self.assertIn("quote", raised.exception.detail.lower())
                self.routes["wgp"].get_model_def.assert_not_called()

    def test_complete_speech_reaches_existing_model_validation(self):
        class ModelLookupReached(Exception):
            pass

        self.routes["wgp"].get_model_def.side_effect = ModelLookupReached
        for name in ("llm_plan_h3_windows", "llm_plan_h3_sequence"):
            with self.subTest(endpoint=name), self.assertRaises(ModelLookupReached):
                asyncio.run(self.routes[name](Request({"prompt": COMPLETE})))

    def test_generation_rejects_bundled_and_imported_h3_before_planning(self):
        for model in ("minimax_h3_ref2va_fused_turbo", "civitai_h3_example_frames"):
            with self.subTest(model=model), self.assertRaises(HTTPError) as raised:
                asyncio.run(self.routes["_prepare_generation_submission"]({
                    "model_type": model, "generation_mode": "video", "prompt": INCOMPLETE,
                }, prepare_only=True))
            self.assertEqual(raised.exception.status_code, 400)
            self.assertIn("quote", raised.exception.detail.lower())

    def test_clear_quote_boundary_accepts_full_corrected_source(self):
        self.routes["_validate_h3_source_quote_boundaries"](COMPLETE)


if __name__ == "__main__":
    unittest.main()
