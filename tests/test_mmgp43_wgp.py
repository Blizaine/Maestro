"""CPU checks for WGP's selected MMGP bootstrap and Profile 0 kwargs."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
_original_sys_path = list(sys.path)
sys.path.insert(0, str(APP))
try:
    from services.memory_runtime import (  # noqa: E402
        EXPERIMENTAL_PROFILE0_RUNTIME,
        LEGACY_RUNTIME,
        _selection_for_backend,
        profile_options,
    )
finally:
    # Importing the pure helper module needs the app root once. Leave path
    # selection to the isolated subprocess contract test below; changing the
    # pytest process path here can make test order decide which MMGP is found.
    sys.path[:] = _original_sys_path


def _extract_wgp_functions(names, namespace):
    source_path = APP / "wgp.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    missing = set(names) - {node.name for node in nodes}
    if missing:
        raise AssertionError(f"Missing WGP helper(s): {sorted(missing)}")
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace


def _profile_options_for(
    backend,
    profile,
    mode="default",
    *,
    preload=0,
    cli_preload=0,
    transformer_budget=0,
    model_type="minimax_h3_ref2va",
    budgets=None,
):
    # The pure option-normalization tests need a value object, not process-wide
    # runtime selection (which intentionally mutates sys.path and enforces the
    # already-imported MMGP identity). Actual selection/import is exercised in
    # its fresh subprocess test below.
    selection = _selection_for_backend(APP, backend)
    config = {
        "read_ahead": False,
        "perc_reserved_mem_max": 0,
        "smart_memory_pinning": False,
        "video_preload_mode": mode,
        "video_preload_in_VRAM": preload,
    }
    definitions = {
        model_type: {"weights": ["initial"]},
        "other-model": {"weights": ["other"]},
    }
    namespace = {
        "args": SimpleNamespace(
            preload=cli_preload,
            transformer_budget=transformer_budget,
            perc_reserved_mem_max=0,
        ),
        "server_config": config,
        "memory_runtime_selection": selection,
        "auto_preload_measures": {},
        "get_model_def": lambda key: definitions.get(key),
        "json": json,
    }
    _extract_wgp_functions(
        ("auto_preload_store", "_configure_runtime_preload", "init_pipe"),
        namespace,
    )
    options = {} if budgets is None else {"budgets": dict(budgets)}
    namespace["init_pipe"]({"transformer": object()}, options, profile, "video")
    namespace["_configure_runtime_preload"](
        options, model_type, profile, "video"
    )
    return selection, config, options, profile_options(selection, profile, options)


class Mmgp43WgpTests(unittest.TestCase):
    def test_bootstrap_selects_before_cuda_memory_and_torch_then_verifies_real_mmgp(self):
        tree = ast.parse((APP / "wgp.py").read_text(encoding="utf-8"))
        top_level = tree.body

        def index(predicate):
            return next(position for position, node in enumerate(top_level) if predicate(node))

        selection_index = index(
            lambda node: isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "memory_runtime_selection"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "select_runtime"
        )
        cuda_import_index = index(
            lambda node: isinstance(node, ast.ImportFrom)
            and node.module == "shared.cuda_memory"
        )
        first_torch_import_index = index(
            lambda node: (
                isinstance(node, ast.Import)
                and any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and (node.module == "torch" or (node.module or "").startswith("torch."))
            )
        )
        real_mmgp_import_index = index(
            lambda node: isinstance(node, ast.Import)
            and any(alias.name == "mmgp" for alias in node.names)
        )
        verify_index = index(
            lambda node: isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "verify_runtime_import"
        )
        self.assertLess(selection_index, cuda_import_index)
        self.assertLess(selection_index, first_torch_import_index)
        self.assertLess(real_mmgp_import_index, verify_index)
        self.assertIn(
            "memory_runtime_config_path = _startup_config_path(sys.argv)",
            (APP / "wgp.py").read_text(encoding="utf-8"),
        )

        load_models = next(
            node
            for node in top_level
            if isinstance(node, ast.FunctionDef) and node.name == "load_models"
        )
        calls = [
            node.func.id
            for node in ast.walk(load_models)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertLess(calls.index("require_model_family"), calls.index("get_runtime_model_def"))
        self.assertLess(calls.index("require_profile_runtime"), calls.index("get_model_filename"))
        source = (APP / "wgp.py").read_text(encoding="utf-8")
        offload_call = source.index("offload.profile(pipe, profile_no=")
        profile_normalizer = source.rfind("kwargs = profile_options(", 0, offload_call)
        self.assertGreater(profile_normalizer, 0)
        self.assertLess(profile_normalizer, offload_call)

    def test_startup_config_path_matches_wgp_default_custom_and_fallback(self):
        namespace = {"os": os}
        _extract_wgp_functions(("_startup_config_path",), namespace)
        resolve = namespace["_startup_config_path"]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            custom = root / "custom config"
            custom.mkdir()
            root_config = root / "wgp_config.json"
            custom_config = custom / "wgp_config.json"
            root_config.write_text('{"memory_runtime":"legacy"}', encoding="utf-8")
            custom_config.write_text(
                '{"memory_runtime":"wangp9905"}', encoding="utf-8"
            )

            self.assertEqual(
                Path(resolve([], cwd=root)),
                root_config,
            )
            self.assertEqual(
                Path(resolve(["--config", str(custom)], cwd=root)),
                custom_config,
            )
            self.assertEqual(
                Path(resolve([f"--config={custom}"], cwd=root)),
                custom_config,
            )
            self.assertEqual(
                Path(resolve(["--config", "custom config"], cwd=root)),
                custom_config,
            )
            self.assertEqual(
                Path(resolve(["--config", str(custom), "--config", "missing"], cwd=root)),
                root_config,
            )

            custom_config.unlink()
            self.assertEqual(
                Path(resolve(["--config", str(custom)], cwd=root)),
                root_config,
            )
            root_config.unlink()
            self.assertEqual(
                Path(resolve(["--config", str(custom)], cwd=root)),
                custom_config,
            )

    def test_profile0_default_dynamic_and_manual_kwargs(self):
        _selection, saved_config, _raw, options = _profile_options_for(
            EXPERIMENTAL_PROFILE0_RUNTIME, 0, "default"
        )
        self.assertEqual(options["autoPreload"], True)
        self.assertIs(options["pinPool"], True)
        self.assertEqual(options["cpuCompute"], "auto")
        self.assertIs(options["cudaGraphs"], False)
        self.assertNotIn("budgets", options)  # profile 0 supplies 1200 / 3000.
        self.assertNotIn("readAhead", options)
        self.assertNotIn("smartPinning", options)
        self.assertIs(saved_config["smart_memory_pinning"], False)

        selection, _config, _raw, dynamic = _profile_options_for(
            EXPERIMENTAL_PROFILE0_RUNTIME, 0, "dynamic"
        )
        self.assertEqual(dynamic["autoPreload"], {})
        self.assertIs(dynamic["pinPool"], True)
        self.assertIs(dynamic["cudaGraphs"], False)

        _selection, _config, _raw, manual_default = _profile_options_for(
            EXPERIMENTAL_PROFILE0_RUNTIME, 0, "manual", preload=0
        )
        self.assertIs(manual_default["autoPreload"], False)
        self.assertEqual(
            manual_default["budgets"],
            {"transformer": 1200, "text_encoder": 100, "*": 3000},
        )

        _selection, _config, _raw, manual_override = _profile_options_for(
            EXPERIMENTAL_PROFILE0_RUNTIME, 0, "manual", preload=2048
        )
        self.assertIs(manual_override["autoPreload"], False)
        self.assertEqual(
            manual_override["budgets"],
            {"transformer": 2048, "text_encoder": 2048, "*": 3000},
        )

        _selection, _config, _raw, cli_manual = _profile_options_for(
            EXPERIMENTAL_PROFILE0_RUNTIME,
            0,
            "default",
            cli_preload=4096,
        )
        self.assertIs(cli_manual["autoPreload"], False)
        self.assertEqual(
            cli_manual["budgets"],
            {"transformer": 4096, "text_encoder": 4096, "*": 4096},
        )

    def test_dynamic_preload_store_stays_model_scoped_and_invalidates_on_definition_change(self):
        definitions = {
            "model-a": {"revision": 1},
            "model-b": {"revision": 1},
        }
        namespace = {
            "auto_preload_measures": {},
            "get_model_def": lambda key: definitions[key],
            "json": json,
        }
        _extract_wgp_functions(("auto_preload_store",), namespace)
        store_a = namespace["auto_preload_store"]("model-a")
        self.assertIs(store_a, namespace["auto_preload_store"]("model-a"))
        self.assertIsNot(store_a, namespace["auto_preload_store"]("model-b"))
        definitions["model-a"] = {"revision": 2}
        self.assertIsNot(store_a, namespace["auto_preload_store"]("model-a"))

    def test_exp_nonzero_profiles_only_enable_auto_preload_when_dynamic(self):
        for profile in (1, 3, 4):
            with self.subTest(profile=profile):
                _selection, _config, _raw, default = _profile_options_for(
                    EXPERIMENTAL_PROFILE0_RUNTIME, profile, "default"
                )
                self.assertIs(default["autoPreload"], False)
                self.assertIs(default["cudaGraphs"], False)

                _selection, _config, _raw, dynamic = _profile_options_for(
                    EXPERIMENTAL_PROFILE0_RUNTIME, profile, "dynamic"
                )
                self.assertEqual(dynamic["autoPreload"], {})
                self.assertIs(dynamic["cudaGraphs"], False)

    def test_legacy_profiles_and_preload_modes_keep_the_previous_kwargs(self):
        _selection, config, raw, profile4 = _profile_options_for(
            LEGACY_RUNTIME, 4, "default"
        )
        self.assertIs(config["smart_memory_pinning"], False)
        self.assertEqual(
            profile4,
            {
                "readAhead": False,
                "perc_reserved_mem_max": 0.0,
                "extraModelsToQuantize": None,
                "budgets": {"transformer": 1200, "text_encoder": 100, "*": 3000},
            },
        )
        self.assertNotIn("autoPreload", raw)

        _selection, _config, _raw, dynamic = _profile_options_for(
            LEGACY_RUNTIME, 3, "dynamic"
        )
        self.assertEqual(dynamic["autoPreload"], {})
        self.assertEqual(dynamic["budgets"], {"*": "70%"})

    def test_profile_options_kwargs_match_real_selected_mmgp_signatures(self):
        script = r'''
import inspect, json, os, sys
sys.path.insert(0, os.environ["MAESTRO_APP_ROOT"])
from services.memory_runtime import select_runtime, verify_runtime_import, profile_options
selection = select_runtime(os.environ["MAESTRO_APP_ROOT"], os.environ["MAESTRO_BACKEND"])
import mmgp
verify_runtime_import(selection, mmgp)
from mmgp import offload
signature = inspect.signature(offload.all)
allowed = set(signature.parameters)
profiles = (0, 1, 3, 4) if selection.backend == "wangp9905" else (1, 3, 4)
for profile in profiles:
    raw = {"readAhead": False, "budgets": {}, "autoPreload": False}
    if selection.backend == "wangp9905":
        raw["cudaGraphs"] = False
    options = profile_options(selection, profile, raw)
    unknown = set(options) - allowed
    assert not unknown, (profile, sorted(unknown), str(signature))
    if selection.backend == "wangp9905":
        assert "readAhead" not in options
        if profile == 0:
            assert "budgets" not in options
            assert options["pinPool"] is True
            assert options["cpuCompute"] == "auto"
            source = inspect.getsource(offload.profile)
            assert 'budgets["transformer"] = 1200' in source
            assert 'budgets["*"] = 3000' in source
        assert options["cudaGraphs"] is False
print(json.dumps({"version": mmgp.__version__, "parameters": sorted(allowed)}))
'''
        for backend in (LEGACY_RUNTIME, EXPERIMENTAL_PROFILE0_RUNTIME):
            environment = os.environ.copy()
            environment.update(
                {
                    "MAESTRO_APP_ROOT": str(APP),
                    "MAESTRO_BACKEND": backend,
                    "CUDA_VISIBLE_DEVICES": "-1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            result = subprocess.run(
                [sys.executable, "-B", "-c", script],
                cwd=APP,
                env=environment,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            with self.subTest(backend=backend):
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                )
                expected_version = "4.0.0" if backend == LEGACY_RUNTIME else "4.4.0"
                self.assertIn(f'"version": "{expected_version}"', result.stdout)


if __name__ == "__main__":
    unittest.main()
