"""Regressions for Classic UI generation-status state."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WGP_PATH = ROOT / "app" / "wgp.py"


def _load_status_helpers() -> dict:
    source = WGP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(WGP_PATH))
    function_names = {
        "initialize_gen_info",
        "get_gen_info",
        "get_generation_status",
        "get_new_refresh_id",
        "merge_status_context",
        "get_latest_status",
        "update_status",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            assigned_names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if assigned_names & {"_GENERATION_STATUS_DEFAULTS", "refresh_id"}:
                selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {}
    exec(compile(module, str(WGP_PATH), "exec"), namespace)
    return namespace


class ClassicUiStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = _load_status_helpers()

    def test_fresh_status_state_has_every_required_field(self):
        gen = self.helpers["initialize_gen_info"]({"queue": []})

        self.assertEqual(0, gen["prompt_no"])
        self.assertEqual(0, gen["prompts_max"])
        self.assertEqual(1, gen["total_generation"])
        self.assertEqual("", gen["progress_status"])

    def test_add_to_queue_status_accepts_legacy_partial_state(self):
        state = {"gen": {"queue": [], "prompts_max": 2}}

        self.helpers["update_status"](state)

        self.assertEqual(0, state["gen"]["prompt_no"])
        self.assertEqual("Prompt 0/2", state["gen"]["progress_status"])
        self.assertGreater(state["gen"]["refresh"], 0)

    def test_missing_gen_dictionary_is_initialized_safely(self):
        state = {}

        status = self.helpers["get_latest_status"](state)

        self.assertEqual("", status)
        self.assertEqual(0, state["gen"]["prompt_no"])


if __name__ == "__main__":
    unittest.main()
