"""CPU-only tests for selecting MMGP before launch imports Torch/allocators."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from services.memory_runtime import (  # noqa: E402
    EXPERIMENTAL_PROFILE0_RUNTIME,
    LEGACY_RUNTIME,
    startup_config_path,
)
from shared.cuda_memory import requested_vram_allocator  # noqa: E402


class MemoryRuntimeBootstrapTests(unittest.TestCase):
    def test_config_resolution_matches_wgp_last_config_and_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "app"
            root.mkdir()
            default = root / "wgp_config.json"
            default.write_text('{"memory_runtime":"legacy"}', encoding="utf-8")
            first = root / "first"
            second = root / "second"
            first.mkdir(); second.mkdir()
            (first / "wgp_config.json").write_text('{"memory_runtime":"legacy"}', encoding="utf-8")
            (second / "wgp_config.json").write_text('{"memory_runtime":"wangp9905"}', encoding="utf-8")

            self.assertEqual(startup_config_path([], cwd=root), str(default))
            self.assertEqual(
                startup_config_path(["--config", str(first), "--config=" + str(second)], cwd=root),
                str(second / "wgp_config.json"),
            )
            # WGP falls back to the app config when the requested config is absent.
            self.assertEqual(
                startup_config_path(["--config", "missing"], cwd=root),
                str(default),
            )
            # An empty/missing value is ignored just as it is by WGP's resolver.
            self.assertEqual(
                startup_config_path(["--config", str(first), "--config"], cwd=root),
                str(first / "wgp_config.json"),
            )

    def test_config_resolution_stays_aligned_with_wgp_startup_parser(self):
        source = (APP / "wgp.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        resolver = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_startup_config_path"
        )
        namespace = {"os": os}
        exec(compile(ast.Module(body=[resolver], type_ignores=[]), "wgp.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "wgp_config.json").write_text("{}", encoding="utf-8")
            custom = root / "custom"
            custom.mkdir()
            (custom / "wgp_config.json").write_text("{}", encoding="utf-8")
            cases = [
                [],
                ["--config", str(custom)],
                ["--config", "missing"],
                ["--config", str(custom), "--config="],
                ["--config", str(custom), "--config"],
            ]
            for argv in cases:
                with self.subTest(argv=argv):
                    self.assertEqual(
                        startup_config_path(argv, cwd=root),
                        namespace["_startup_config_path"](argv, cwd=str(root)),
                    )

    def test_vram_allocator_cli_override_semantics_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "wgp_config.json"
            config.write_text('{"vram_allocator":"default"}', encoding="utf-8")
            self.assertEqual(
                requested_vram_allocator(
                    ["--vram-allocator", "vmm_spill", "--vram-allocator=vmm"],
                    str(config),
                ),
                "vmm_spill",
            )

    def test_launch_source_selects_runtime_before_cuda_allocator_and_torch(self):
        source = (APP / "launch.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level = tree.body
        select_call_index = next(
            index for index, node in enumerate(top_level)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_memory_runtime_selection" for target in node.targets)
        )
        allocator_import_index = next(
            index for index, node in enumerate(top_level)
            if isinstance(node, ast.ImportFrom) and node.module == "shared.cuda_memory"
        )
        allocator_apply_index = next(
            index for index, node in enumerate(top_level)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "apply_startup_settings"
        )
        torch_import_index = next(
            index for index, node in enumerate(top_level)
            if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
        )
        self.assertLess(select_call_index, allocator_import_index)
        self.assertLess(allocator_import_index, allocator_apply_index)
        self.assertLess(allocator_apply_index, torch_import_index)
        self.assertIn("startup_config_path(sys.argv, cwd=_app_dir)", source)

    @staticmethod
    def _make_fake_app(root: Path) -> Path:
        app = root / "app"
        app.mkdir()
        # Exercise the actual launch bootstrap block while stubbing only the
        # side-effecting allocator/Torch/MMGP imports.
        launch_source = (APP / "launch.py").read_text(encoding="utf-8")
        start = launch_source.index("# --- Bootstrap: CWD must be app/ and sys.argv must be patched before importing wgp ---")
        end = launch_source.index("# Preserve original argv, patch for wgp's argparse", start)
        bootstrap = "import gc\nimport sys\nimport os\n" + launch_source[start:end]
        (app / "launch.py").write_text(bootstrap, encoding="utf-8")

        services = app / "services"
        services.mkdir()
        (services / "__init__.py").write_text("", encoding="utf-8")
        (services / "memory_runtime.py").write_text(
            (APP / "services" / "memory_runtime.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        shared = app / "shared"
        shared.mkdir()
        (shared / "__init__.py").write_text("", encoding="utf-8")
        (app / "runtime_probe.py").write_text(
            "import json, os\n"
            "def record(event):\n"
            "    path = os.environ['MEMORY_BOOT_PROBE']\n"
            "    try:\n"
            "        with open(path, encoding='utf-8') as reader: events = json.load(reader)\n"
            "    except FileNotFoundError: events = []\n"
            "    events.append(event)\n"
            "    with open(path, 'w', encoding='utf-8') as writer: json.dump(events, writer)\n""",
            encoding="utf-8",
        )
        (shared / "cuda_memory.py").write_text(
            "import os, sys\n"
            "from pathlib import Path\n"
            "from runtime_probe import record\n"
            "def apply_startup_settings(argv, config_filename):\n"
            "    record({'stage': 'allocator-enter', 'argv': list(argv), 'config_filename': config_filename})\n"
            "    import torch\n"
            "    import mmgp\n"
            "    from services.memory_runtime import read_saved_runtime, select_runtime, startup_config_path, verify_runtime_import\n"
            "    app_dir = Path(__file__).resolve().parents[1]\n"
            "    saved = read_saved_runtime(startup_config_path(argv, cwd=app_dir))\n"
            "    selection = select_runtime(app_dir, saved)\n"
            "    verify_runtime_import(selection, mmgp)\n"
            "    record({'stage': 'same-runtime-reselect', 'backend': selection.backend, 'mmgp_path': mmgp.__file__})\n"
            "    record({'stage': 'allocator-return'})\n""",
            encoding="utf-8",
        )
        (app / "torch.py").write_text(
            "import sys\n"
            "from runtime_probe import record\n"
            "record({'stage': 'torch', 'mmgp_loaded': 'mmgp' in sys.modules, 'path0': sys.path[0]})\n""",
            encoding="utf-8",
        )
        for backend, directory, version in (
            (LEGACY_RUNTIME, app / "mmgp", "4.0.0"),
            (EXPERIMENTAL_PROFILE0_RUNTIME, app / "mmgp_backends" / "v43" / "mmgp", "4.4.0"),
        ):
            directory.mkdir(parents=True)
            (directory / "__init__.py").write_text(
                f"__version__ = {version!r}\n"
                "from runtime_probe import record\n"
                f"record({{'stage': 'mmgp', 'backend': {backend!r}, 'path': __file__}})\n",
                encoding="utf-8",
            )
        return app

    @staticmethod
    def _run_fake_bootstrap(app: Path, args: list[str]):
        probe = app / "probe.json"
        env = os.environ.copy()
        env["MEMORY_BOOT_PROBE"] = str(probe)
        result = subprocess.run(
            [sys.executable, str(app / "launch.py"), *args],
            cwd=app.parent,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        events = json.loads(probe.read_text(encoding="utf-8")) if probe.exists() else []
        return result, events

    def test_default_legacy_runtime_is_selected_before_torch_and_reselection_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._make_fake_app(Path(temporary))
            result, events = self._run_fake_bootstrap(app, [])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([event["stage"] for event in events], [
                "allocator-enter", "torch", "mmgp", "same-runtime-reselect", "allocator-return",
            ])
            self.assertFalse(events[1]["mmgp_loaded"])
            self.assertEqual(events[1]["path0"], str(app))
            self.assertEqual(events[2]["backend"], LEGACY_RUNTIME)
            self.assertEqual(events[2]["path"], str(app / "mmgp" / "__init__.py"))

    def test_last_custom_config_selects_experimental_before_allocator_and_preserves_cli_args(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._make_fake_app(Path(temporary))
            first = app / "cfg-first"
            last = app / "cfg-last"
            first.mkdir(); last.mkdir()
            (first / "wgp_config.json").write_text('{"memory_runtime":"legacy"}', encoding="utf-8")
            (last / "wgp_config.json").write_text('{"memory_runtime":"wangp9905"}', encoding="utf-8")
            args = [
                "--config", str(first), "--config=" + str(last),
                "--vram-allocator", "vmm_spill", "--ram-allocator=mmgp",
            ]
            result, events = self._run_fake_bootstrap(app, args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(events[0]["argv"], [str(app / "launch.py"), *args])
            self.assertEqual(events[0]["config_filename"], str(app / "wgp_config.json"))
            self.assertEqual(events[1]["stage"], "torch")
            self.assertEqual(Path(events[1]["path0"]).resolve(), app / "mmgp_backends" / "v43")
            self.assertEqual(events[2]["backend"], EXPERIMENTAL_PROFILE0_RUNTIME)
            self.assertEqual(events[2]["path"], str(app / "mmgp_backends" / "v43" / "mmgp" / "__init__.py"))
            self.assertEqual(events[3]["stage"], "same-runtime-reselect")

    def test_missing_custom_config_falls_back_to_app_config_before_any_backend_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._make_fake_app(Path(temporary))
            (app / "wgp_config.json").write_text('{"memory_runtime":"wangp9905"}', encoding="utf-8")
            result, events = self._run_fake_bootstrap(app, ["--config", "missing-custom-directory"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(events[1]["stage"], "torch")
            self.assertEqual(Path(events[1]["path0"]).resolve(), app / "mmgp_backends" / "v43")
            self.assertEqual(events[2]["backend"], EXPERIMENTAL_PROFILE0_RUNTIME)

    def test_invalid_saved_runtime_fails_before_torch_or_allocator_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self._make_fake_app(Path(temporary))
            (app / "wgp_config.json").write_text('{"memory_runtime":"almost-experimental"}', encoding="utf-8")
            result, events = self._run_fake_bootstrap(app, [])
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(events, [])
            self.assertIn("Invalid memory_runtime", result.stderr)


if __name__ == "__main__":
    unittest.main()
