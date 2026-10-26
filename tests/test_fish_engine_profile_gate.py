"""Regression tests for the model-aware LM engine profile gate in wgp.py.

The test compiles and executes the capability assignment and actual
profile-gating If statement from wgp.py as a small function. This avoids
importing the full UI/runtime module while exercising the production branch.
"""

from __future__ import annotations

import ast
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WGP_PATH = ROOT / "app" / "wgp.py"


def build_profile_gate():
    source = WGP_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(WGP_PATH))
    candidates = []
    for node in ast.walk(module):
        if not isinstance(node, ast.If):
            continue
        condition_names = {
            child.id
            for child in ast.walk(node.test)
            if isinstance(child, ast.Name)
        }
        if {"lm_decoder_engine_obtained", "profile"} <= condition_names:
            candidates.append(node)
    if len(candidates) != 1:
        raise AssertionError(
            f"expected one LM engine profile gate in {WGP_PATH}, found {len(candidates)}"
        )
    gate_if = candidates[0]
    support_assignment = next(
        (
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and isinstance(target.ctx, ast.Store)
                and target.id == "lm_engine_supports_profile_offload"
                for target in ast.walk(node)
            )
        ),
        None,
    )
    if support_assignment is None:
        raise AssertionError(
            f"expected the explicit offload capability assignment in {WGP_PATH}"
        )

    gate_block = None
    for parent in ast.walk(module):
        for _, value in ast.iter_fields(parent):
            if not isinstance(value, list) or gate_if not in value:
                continue
            gate_index = value.index(gate_if)
            if (
                gate_index > 0
                and value[gate_index - 1] is support_assignment
            ):
                gate_block = [support_assignment, gate_if]
                break
        if gate_block is not None:
            break
    if gate_block is None:
        raise AssertionError(
            f"the explicit offload assignment must immediately precede the gate in {WGP_PATH}"
        )

    function = ast.FunctionDef(
        name="apply_profile_gate",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="lm_decoder_engine_obtained"),
                ast.arg(arg="profile"),
                ast.arg(arg="model_def"),
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[
            *gate_block,
            ast.Return(value=ast.Name(id="lm_decoder_engine_obtained", ctx=ast.Load())),
        ],
        decorator_list=[],
    )
    test_module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    namespace = {}
    # This function contains only the selected production If statement.
    exec(compile(test_module, str(WGP_PATH), "exec"), namespace)  # noqa: S102
    return namespace["apply_profile_gate"]


class FishEngineProfileGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.apply_profile_gate = staticmethod(build_profile_gate())

    def run_gate(self, profile, engine, model_def=None):
        with redirect_stdout(StringIO()):
            return self.apply_profile_gate(
                lm_decoder_engine_obtained=engine,
                profile=profile,
                model_def={} if model_def is None else model_def,
            )

    def test_vllm_offload_capability_keeps_profiles_2_4_and_5(self):
        for profile in (2, 4, 5):
            with self.subTest(profile=profile):
                self.assertEqual(
                    self.run_gate(
                        profile,
                        "vllm",
                        {"lm_engine_supports_offload": True},
                    ),
                    "vllm",
                )

    def test_vllm_without_explicit_offload_capability_still_downgrades(self):
        for profile in (2, 4, 5):
            for model_def in (
                {"lm_engine_supports_offload": False},
                {},
            ):
                with self.subTest(profile=profile, model_def=model_def):
                    self.assertEqual(self.run_gate(profile, "vllm", model_def), "legacy")

    def test_profiles_1_and_3_keep_the_requested_engine(self):
        for profile in (1, 3):
            for engine in ("vllm", "cg"):
                for model_def in (
                    {"lm_engine_supports_offload": True},
                    {"lm_engine_supports_offload": False},
                    {},
                ):
                    with self.subTest(profile=profile, engine=engine, model_def=model_def):
                        self.assertEqual(
                            self.run_gate(profile, engine, model_def),
                            engine,
                        )

    def test_whole_graph_keeps_the_full_bank_profile_restriction(self):
        for profile in (2, 4, 5):
            with self.subTest(profile=profile):
                self.assertEqual(
                    self.run_gate(
                        profile,
                        "cg",
                        {"lm_engine_supports_offload": True},
                    ),
                    "legacy",
                )

    def test_legacy_engine_is_unchanged(self):
        for profile in (1, 2, 3, 4, 5):
            with self.subTest(profile=profile):
                self.assertEqual(self.run_gate(profile, "legacy"), "legacy")


if __name__ == "__main__":
    unittest.main()
