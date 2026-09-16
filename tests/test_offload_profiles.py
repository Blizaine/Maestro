"""Low-memory profile variants must retain WanGP's base profile budgets."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


def init_pipe(preload=0, saved_preload=0, transformer_budget=None):
    path = Path(__file__).resolve().parents[1] / "app" / "wgp.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "init_pipe")
    namespace = {"args": SimpleNamespace(preload=preload), "server_config": {"preload_in_VRAM": saved_preload}}
    if transformer_budget is not None:
        namespace["args"].transformer_budget = transformer_budget
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["init_pipe"]


class OffloadProfileTests(unittest.TestCase):
    def test_residency_override_changes_only_the_first_transformer(self):
        for profile in (2, 4, 4.5, 5):
            with self.subTest(profile=profile):
                options = {
                    "budgets": {"transformer": 200, "transformer2": 300, "vae": 512},
                    "workingVRAM": {"transformer": 10240},
                }
                init_pipe(transformer_budget=9500)(
                    {"transformer": object(), "transformer2": object()}, options, profile
                )
                self.assertEqual(options["budgets"], {
                    "transformer": 9500, "transformer2": 300, "vae": 512,
                    "text_encoder": 100, "*": 1000 if profile == 5 else 3000,
                })
                self.assertEqual(options["workingVRAM"], {"transformer": 10240})

    def test_manual_preload_wins_over_job_residency(self):
        for preload, saved_preload, expected in ((5000, 0, 5000), (0, 6000, 6000), (5000, 6000, 5000)):
            with self.subTest(preload=preload, saved_preload=saved_preload):
                options = {}
                init_pipe(preload, saved_preload, transformer_budget=9500)(
                    {"transformer": object()}, options, 5
                )
                self.assertEqual(options["budgets"], {
                    "transformer": expected, "text_encoder": expected, "*": expected,
                })

    def test_residency_override_does_not_change_full_residency_profiles(self):
        for profile in (1, 3, 3.5):
            with self.subTest(profile=profile):
                expected, actual = {}, {}
                init_pipe()({"transformer": object()}, expected, profile)
                init_pipe(transformer_budget=9500)({"transformer": object()}, actual, profile)
                self.assertEqual(actual, expected)

    def test_nonpositive_budget_keeps_model_default(self):
        for budget in (0, -1):
            with self.subTest(budget=budget):
                options = {"budgets": {"transformer": 200}}
                init_pipe(transformer_budget=budget)({"transformer": object()}, options, 5)
                self.assertEqual(options["budgets"]["transformer"], 200)

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
