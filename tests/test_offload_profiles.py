"""Low-memory profile variants must retain WanGP's base profile budgets."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


def init_pipe(preload=0, saved_preload=0):
    path = Path(__file__).resolve().parents[1] / "app" / "wgp.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "init_pipe")
    namespace = {"args": SimpleNamespace(preload=preload), "server_config": {"preload_in_VRAM": saved_preload}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["init_pipe"]


class OffloadProfileTests(unittest.TestCase):
    def test_profile_3_5_keeps_70_percent_budget_without_pinning(self):
        options = {}
        self.assertEqual(init_pipe()({"transformer": object()}, options, 3.5), 3)
        self.assertEqual(options["budgets"], {"*": "70%"})
        self.assertIs(options["pinnedMemory"], False)

    def test_profile_4_5_keeps_streaming_limits_without_async_transfers(self):
        options = {"budgets": {"transformer": 200, "transformer2": 300, "vae": 512}}
        self.assertEqual(init_pipe()({"transformer": object(), "transformer2": object()}, options, 4.5), 4)
        self.assertEqual(options["budgets"], {"transformer": 200, "transformer2": 300,
                                              "text_encoder": 100, "*": 3000, "vae": 512})
        self.assertIs(options["asyncTransfers"], False)

    def test_preload_override_still_applies_to_streaming_variants(self):
        for fn in (init_pipe(preload=640), init_pipe(saved_preload=640)):
            options = {}
            fn({"transformer": object()}, options, 4.5)
            self.assertEqual(options["budgets"], {"transformer": 640, "text_encoder": 640, "*": 3000})

    def test_integer_profiles_keep_their_previous_policy(self):
        for profile in (1, 2, 3, 4, 5):
            with self.subTest(profile=profile):
                options = {}
                self.assertEqual(init_pipe()({"transformer": object()}, options, profile), profile)
                expected = {} if profile == 1 else ({"*": "70%"} if profile == 3 else
                    {"transformer": 100, "text_encoder": 100, "*": 1000 if profile == 5 else 3000})
                self.assertEqual(options["budgets"], expected)


if __name__ == "__main__":
    unittest.main()
