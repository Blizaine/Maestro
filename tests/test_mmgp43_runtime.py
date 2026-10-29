"""CPU-only checks for the staged MMGP runtime selector and source snapshot."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from services.memory_runtime import (  # noqa: E402
    EXPERIMENTAL_PROFILE0_RUNTIME,
    LEGACY_RUNTIME,
    MemoryRuntimeError,
    RuntimeRestartRequired,
    active_runtime_selection,
    allocator_asset_status,
    graph_policy,
    profile_options,
    require_model_family,
    require_profile,
    require_profile_runtime,
    read_saved_runtime,
    runtime_snapshot,
    select_runtime,
    validate_runtime_updates,
    validate_system_memory_updates,
    verify_runtime_import,
)


def write_vmm_fixture(directory: Path, content: bytes) -> None:
    """Exercise static presence gates on either supported CI host, never load a library."""
    for name in ("vmm_alloc_win_amd64.dll", "vmm_alloc_linux_x86_64.so"):
        (directory / name).write_bytes(content)


class IsolatedMmgpImportTests(unittest.TestCase):
    def setUp(self):
        self.saved_path = list(sys.path)
        self.saved_mmgp_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == "mmgp" or name.startswith("mmgp.")
        }
        for name in self.saved_mmgp_modules:
            sys.modules.pop(name, None)

    def tearDown(self):
        for name in tuple(sys.modules):
            if name == "mmgp" or name.startswith("mmgp."):
                sys.modules.pop(name, None)
        sys.modules.update(self.saved_mmgp_modules)
        sys.path[:] = self.saved_path


class Mmgp43RuntimeTests(IsolatedMmgpImportTests):
    def make_app(self, root: Path) -> Path:
        app = root / "app"
        for package, version in (
            (app / "mmgp", "4.0.0"),
            (app / "mmgp_backends" / "v43" / "mmgp", "4.4.0"),
        ):
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                f"__version__ = {version!r}\n", encoding="utf-8"
            )
        return app

    def test_read_saved_runtime_is_strict_read_only_and_defaults_only_when_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "server.json"
            self.assertEqual(read_saved_runtime(config_path), LEGACY_RUNTIME)

            config_path.write_text('{"other_setting": true}\n', encoding="utf-8")
            original = config_path.read_text(encoding="utf-8")
            self.assertEqual(read_saved_runtime(config_path), LEGACY_RUNTIME)
            self.assertEqual(config_path.read_text(encoding="utf-8"), original)

            for value in (LEGACY_RUNTIME, EXPERIMENTAL_PROFILE0_RUNTIME):
                config_path.write_text(
                    json.dumps({"memory_runtime": value, "untouched": [1, 2]}),
                    encoding="utf-8",
                )
                original = config_path.read_bytes()
                self.assertEqual(read_saved_runtime(config_path), value)
                self.assertEqual(config_path.read_bytes(), original)

            for value in (None, "", "Legacy", "wangp9905 ", 0, False):
                config_path.write_text(
                    json.dumps({"memory_runtime": value}), encoding="utf-8"
                )
                with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "Invalid memory_runtime"
                ):
                    read_saved_runtime(config_path)

            config_path.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(MemoryRuntimeError, "Could not read"):
                read_saved_runtime(config_path)

    def test_read_saved_runtime_does_not_import_torch_or_mmgp(self):
        script = r'''
import os
import sys
sys.path.insert(0, os.environ["MAESTRO_APP_ROOT"])
from services.memory_runtime import read_saved_runtime
assert read_saved_runtime(os.environ["MAESTRO_CONFIG_PATH"]) == "legacy"
assert "torch" not in sys.modules
assert "mmgp" not in sys.modules
print("saved runtime read without torch/mmgp")
'''
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "server.json"
            config_path.write_text("{}", encoding="utf-8")
            environment = os.environ.copy()
            environment["MAESTRO_APP_ROOT"] = str(APP)
            environment["MAESTRO_CONFIG_PATH"] = str(config_path)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [sys.executable, "-B", "-c", script],
                cwd=APP,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("without torch/mmgp", result.stdout)

    def test_runtime_snapshot_separates_saved_and_active_capabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.make_app(Path(temporary))
            active = select_runtime(app, LEGACY_RUNTIME)
            cfg = {
                "memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME,
                "vram_allocator": "vmm",
                "video_profile": 0,
                "image_profile": 4,
                "audio_profile": 5,
            }
            before = dict(cfg)
            snapshot = runtime_snapshot(active, cfg)
            self.assertEqual(snapshot["saved_runtime"], EXPERIMENTAL_PROFILE0_RUNTIME)
            self.assertEqual(snapshot["active_runtime"], LEGACY_RUNTIME)
            self.assertTrue(snapshot["restart_required"])
            self.assertEqual(
                snapshot["selected_label"], "Experimental Profile0 (WanGP9905)"
            )
            self.assertFalse(snapshot["profile0_native_available"])
            self.assertIn("absent", snapshot["profile0_capability_reason"])
            self.assertFalse(snapshot["fish_supported"])
            self.assertTrue(snapshot["active_fish_supported"])
            self.assertEqual(cfg, before)

            experimental = select_runtime(app, EXPERIMENTAL_PROFILE0_RUNTIME)
            allocator_dir = experimental.package_dir / "allocator"
            allocator_dir.mkdir(parents=True)
            write_vmm_fixture(allocator_dir, b"")
            empty_snapshot = runtime_snapshot(experimental, cfg)
            self.assertFalse(empty_snapshot["profile0_native_available"])
            self.assertFalse(empty_snapshot["experimental_runtime_selectable"])
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                validate_system_memory_updates(
                    {"memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME}, cfg, active
                )
            # Non-empty fixture bytes exercise only the static asset gate; the
            # file is never loaded or treated as an ABI/GPU validation result.
            write_vmm_fixture(allocator_dir, b"fixture-only")
            snapshot = runtime_snapshot(experimental, cfg)
            self.assertTrue(snapshot["profile0_native_available"])
            self.assertTrue(snapshot["experimental_runtime_selectable"])
            self.assertFalse(snapshot["profile0_native_abi_verified"])
            self.assertIn("file presence only", snapshot["profile0_native_validation_reason"])
            self.assertTrue(snapshot["profile0_saved_configuration_eligible"])
            self.assertIn(
                "validate the VMM binary ABI", snapshot["profile0_capability_reason"]
            )
            self.assertFalse(snapshot["restart_required"])
            self.assertFalse(snapshot["fish_supported"])
            self.assertFalse(snapshot["active_fish_supported"])
            self.assertFalse(snapshot["profile0_ram_native_available"])
            self.assertIn("ram_alloc_", snapshot["profile0_ram_capability_reason"])

    def test_memory_runtime_preflight_rejects_unsupported_saved_choices_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.make_app(Path(temporary))
            active = select_runtime(app, LEGACY_RUNTIME)
            cfg = {
                "memory_runtime": LEGACY_RUNTIME,
                "video_profile": 3,
                "image_profile": 4.5,
                "audio_profile": 5,
                "vram_allocator": "default",
                "ram_allocator": "default",
                "read_ahead": False,
                "smart_memory_pinning": False,
                "video_preload_mode": "manual",
                "video_preload_in_VRAM": 2048,
            }
            before = dict(cfg)
            with self.assertRaisesRegex(ValueError, "cannot be selected or saved"):
                validate_system_memory_updates(
                    {"memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME}, cfg, active
                )
            self.assertEqual(cfg, before)

            allocator_dir = app / "mmgp_backends" / "v43" / "mmgp" / "allocator"
            allocator_dir.mkdir(parents=True)
            write_vmm_fixture(allocator_dir, b"fixture-only")
            for body, message in (
                (
                    {"memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME, "read_ahead": True},
                    "removed read-ahead",
                ),
                (
                    {"memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME, "ram_allocator": "mmgp"},
                    "companion binary",
                ),
                ({"memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME, "video_profile": 0}, "requires vram_allocator"),
            ):
                with self.subTest(body=body), self.assertRaisesRegex(ValueError, message):
                    validate_system_memory_updates(body, cfg, active)
                self.assertEqual(cfg, before)
            # Selecting the experimental package alone preserves all legacy
            # profile and hand-tuned memory values; no migration is returned.
            self.assertEqual(
                validate_system_memory_updates(
                    {"memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME}, cfg, active
                ),
                ({}, {"memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME}),
            )
            self.assertEqual(cfg, before)

    def test_active_selection_is_inferred_only_from_the_loaded_module_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.make_app(Path(temporary))
            self.assertIsNone(active_runtime_selection(app))
            selected = select_runtime(app, EXPERIMENTAL_PROFILE0_RUNTIME)
            import mmgp

            self.assertEqual(active_runtime_selection(app), selected)
            self.assertEqual(active_runtime_selection(app, selected), selected)
            self.assertEqual(Path(mmgp.__file__).resolve(), selected.package_dir / "__init__.py")

    def test_preload_identity_hashes_imported_source_and_keeps_legacy_raw_hash(self):
        from services.performance_tuning import _auto_preload_runtime_identity

        legacy = ModuleType("mmgp")
        legacy.__file__ = str(APP / "mmgp" / "__init__.py")
        legacy.__version__ = "4.0.0"
        sys.modules["mmgp"] = legacy
        legacy_identity = _auto_preload_runtime_identity()
        expected_legacy = hashlib.sha256((APP / "mmgp" / "auto_preload.py").read_bytes()).hexdigest()
        self.assertEqual(legacy_identity, expected_legacy)

        experimental = ModuleType("mmgp")
        experimental.__file__ = str(APP / "mmgp_backends" / "v43" / "mmgp" / "__init__.py")
        experimental.__version__ = "4.4.0"
        sys.modules["mmgp"] = experimental
        experimental_identity = _auto_preload_runtime_identity()
        self.assertNotEqual(experimental_identity, legacy_identity)
        self.assertEqual(len(experimental_identity), 64)

    def test_validate_runtime_updates_is_pure_atomic_and_preserves_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.make_app(Path(temporary))
            active = select_runtime(app, LEGACY_RUNTIME)
            experimental = select_runtime(app, EXPERIMENTAL_PROFILE0_RUNTIME)
            allocator_dir = experimental.package_dir / "allocator"
            allocator_dir.mkdir(parents=True)
            write_vmm_fixture(allocator_dir, b"fixture-only")

            cfg = {
                "memory_runtime": LEGACY_RUNTIME,
                "video_profile": 3.5,
                "image_profile": 5,
                "audio_profile": 4.5,
                "vram_allocator": "default",
            }
            body = {
                "memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME,
                "video_profile": 0,
                "vram_allocator": "vmm",
            }
            cfg_before = dict(cfg)
            body_before = dict(body)
            validated = validate_runtime_updates(body, cfg, active)
            self.assertEqual(validated, body)
            self.assertEqual(cfg, cfg_before)
            self.assertEqual(body, body_before)
            # Updating the runtime leaves every existing profile untouched.
            self.assertEqual(cfg["video_profile"], 3.5)
            self.assertEqual(cfg["image_profile"], 5)
            self.assertEqual(cfg["audio_profile"], 4.5)

            for profile in (0, 1, 2, 3, 3.5, 4, 4.5, 5):
                with self.subTest(profile=profile):
                    self.assertEqual(
                        validate_runtime_updates(
                            {
                                "memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME,
                                "vram_allocator": "vmm",
                                "video_profile": profile,
                            },
                            cfg,
                            active,
                        )["video_profile"],
                        profile,
                    )

            invalid_body = {
                "memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME,
                "video_profile": 0,
                "image_profile": 9,
                "vram_allocator": "vmm",
            }
            invalid_before = dict(invalid_body)
            with self.assertRaisesRegex(ValueError, "valid choices"):
                validate_runtime_updates(invalid_body, cfg, active)
            self.assertEqual(cfg, cfg_before)
            self.assertEqual(invalid_body, invalid_before)

            for update in (
                {"video_profile": 0},
                {"memory_runtime": "unknown"},
                {"vram_allocator": "bad"},
            ):
                with self.subTest(update=update), self.assertRaises(ValueError):
                    validate_runtime_updates(update, cfg, active)
                self.assertEqual(cfg, cfg_before)

            with self.assertRaisesRegex(ValueError, "requires memory_runtime"):
                validate_runtime_updates(
                    {"video_profile": 0, "vram_allocator": "vmm"}, cfg, active
                )
            with self.assertRaisesRegex(ValueError, "requires vram_allocator"):
                validate_runtime_updates(
                    {
                        "memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME,
                        "video_profile": 0,
                    },
                    cfg,
                    active,
                )

            # The runtime guard checks the active allocator even if saved
            # settings passed the static presence check. It also rejects
            # Profile 0 while the legacy backend is still active.
            with self.assertRaisesRegex(MemoryRuntimeError, "requires"):
                require_profile_runtime(
                    active,
                    0,
                    allocator_active="vmm",
                    platform="win32",
                    machine="AMD64",
                )
            with self.assertRaisesRegex(MemoryRuntimeError, "active"):
                require_profile_runtime(
                    experimental,
                    0,
                    allocator_active=None,
                    platform="win32",
                    machine="AMD64",
                )
            require_profile_runtime(
                experimental,
                0,
                allocator_active="vmm",
                platform="win32",
                machine="AMD64",
            )

    def test_missing_preference_keeps_legacy_and_imports_from_app(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.make_app(Path(temporary))
            selected = select_runtime(app)
            self.assertEqual(selected.backend, LEGACY_RUNTIME)
            self.assertEqual(Path(sys.path[0]), app.resolve())
            import mmgp

            self.assertEqual(Path(mmgp.__file__).resolve(), app / "mmgp" / "__init__.py")
            self.assertEqual(verify_runtime_import(selected), "4.0.0")

    def test_explicit_experimental_choice_puts_its_package_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.make_app(Path(temporary))
            selected = select_runtime(app, EXPERIMENTAL_PROFILE0_RUNTIME)
            self.assertEqual(selected.backend, EXPERIMENTAL_PROFILE0_RUNTIME)
            self.assertEqual(
                Path(sys.path[0]), app / "mmgp_backends" / "v43"
            )
            self.assertEqual(Path(sys.path[1]), app.resolve())
            import mmgp

            self.assertEqual(
                Path(mmgp.__file__).resolve(),
                app / "mmgp_backends" / "v43" / "mmgp" / "__init__.py",
            )
            self.assertEqual(verify_runtime_import(selected), "4.4.0")

    def test_backend_change_after_import_requires_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.make_app(Path(temporary))
            select_runtime(app, LEGACY_RUNTIME)
            import mmgp

            self.assertEqual(mmgp.__version__, "4.0.0")
            with self.assertRaisesRegex(RuntimeRestartRequired, "restart"):
                select_runtime(app, EXPERIMENTAL_PROFILE0_RUNTIME)

    def test_profile_numbers_and_manual_options_are_never_remapped(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.make_app(Path(temporary))
            legacy = select_runtime(app)
            for value in (1, 2, 3, 3.5, 4, 4.5, 5):
                self.assertIs(require_profile(legacy, value), value)
                manual = {"budgets": {"transformer": 1800}, "pinnedMemory": False}
                self.assertEqual(profile_options(legacy, value, manual), manual)
            with self.assertRaisesRegex(MemoryRuntimeError, "requires"):
                require_profile(legacy, 0)

            experimental = select_runtime(app, EXPERIMENTAL_PROFILE0_RUNTIME)
            self.assertEqual(
                profile_options(experimental, 0),
                {
                    "pinPool": True,
                    "cpuCompute": "auto",
                    "autoPreload": True,
                    "cudaGraphs": False,
                },
            )
            overrides = {
                "budgets": {"transformer": 1500},
                "pinnedMemory": False,
                "autoPreload": False,
            }
            self.assertEqual(
                profile_options(experimental, 0, overrides),
                {
                    "pinPool": True,
                    "cpuCompute": "auto",
                    "autoPreload": False,
                    "cudaGraphs": False,
                    **overrides,
                },
            )
            with self.assertRaisesRegex(ValueError, "valid choices"):
                require_profile(experimental, 6)

    def test_profile0_adapter_drops_removed_disabled_kwarg_and_empty_budget_sentinel(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.make_app(Path(temporary))
            experimental = select_runtime(app, EXPERIMENTAL_PROFILE0_RUNTIME)
            adapted = profile_options(
                experimental,
                0,
                {"readAhead": False, "budgets": {}, "perc_reserved_mem_max": 0.35},
            )
            self.assertNotIn("readAhead", adapted)
            self.assertNotIn("budgets", adapted)
            self.assertEqual(adapted["perc_reserved_mem_max"], 0.35)
            self.assertEqual(adapted["pinPool"], True)
            self.assertEqual(adapted["cpuCompute"], "auto")
            self.assertEqual(adapted["autoPreload"], True)

            manual_budgets = {"transformer": 800, "*": 2400}
            adapted = profile_options(
                experimental,
                0,
                {"budgets": manual_budgets, "readAhead": False},
            )
            self.assertEqual(adapted["budgets"], manual_budgets)

            with self.assertRaisesRegex(MemoryRuntimeError, "removed MMGP's readAhead"):
                profile_options(experimental, 0, {"readAhead": True})

            nonzero = profile_options(
                experimental,
                4,
                {"readAhead": False, "budgets": {}, "cudaGraphs": False},
            )
            self.assertNotIn("readAhead", nonzero)
            self.assertNotIn("budgets", nonzero)
            with self.assertRaisesRegex(MemoryRuntimeError, "removed MMGP's readAhead"):
                profile_options(experimental, 4, {"readAhead": True})

            # Legacy keeps its existing read-ahead and graph paths unchanged.
            legacy = select_runtime(app, LEGACY_RUNTIME)
            legacy_options = profile_options(
                legacy, 4, {"readAhead": True, "cudaGraphs": True}
            )
            self.assertTrue(legacy_options["readAhead"])
            self.assertTrue(legacy_options["cudaGraphs"])

    def test_allocator_and_graph_guards_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.make_app(Path(temporary))
            experimental = select_runtime(app, EXPERIMENTAL_PROFILE0_RUNTIME)
            status = allocator_asset_status(
                experimental, platform="win32", machine="AMD64"
            )
            self.assertFalse(status.present)
            self.assertEqual(status.path.name, "vmm_alloc_win_amd64.dll")
            with self.assertRaisesRegex(MemoryRuntimeError, "VMM allocator"):
                require_profile_runtime(
                    experimental,
                    0,
                    allocator_active=None,
                    platform="win32",
                    machine="AMD64",
                )
            with self.assertRaisesRegex(MemoryRuntimeError, "graph capture"):
                profile_options(experimental, 0, {"cudaGraphs": True})
            self.assertFalse(graph_policy(experimental, 0))
            with self.assertRaisesRegex(MemoryRuntimeError, "not enabled"):
                graph_policy(experimental, 1, requested=True)
            self.assertTrue(graph_policy(select_runtime(app), 1, requested=True))
            with self.assertRaisesRegex(MemoryRuntimeError, "only for profiles 1 and 3"):
                graph_policy(select_runtime(app), 4, requested=True)

    def test_fish_is_explicitly_unsupported_on_new_expert_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.make_app(Path(temporary))
            experimental = select_runtime(app, EXPERIMENTAL_PROFILE0_RUNTIME)
            with self.assertRaisesRegex(MemoryRuntimeError, "Fish Audio adapter"):
                require_model_family(experimental, "fish_audio")
            require_model_family(experimental, "wan_video")

    def test_real_wgp_app_path_keeps_legacy_import_as_default(self):
        script = r'''
import os
import sys
app = os.environ["MAESTRO_APP_ROOT"]
sys.path.insert(0, app)  # wgp.py startup path setup before its MMGP imports
from services.memory_runtime import select_runtime, verify_runtime_import
selection = select_runtime(app)
import mmgp
assert verify_runtime_import(selection) == "4.0.0"
assert os.path.normcase(os.path.abspath(mmgp.__file__)).startswith(os.path.normcase(os.path.abspath(app)))
print("default MMGP import:", mmgp.__file__, mmgp.__version__)
'''
        environment = os.environ.copy()
        environment["MAESTRO_APP_ROOT"] = str(APP)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=APP,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("default MMGP import:", result.stdout)

    def test_real_app_startup_resolution_imports_every_upstream_module_and_cleans_up(self):
        script = r'''
import importlib
import os
import pkgutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

app = os.environ["MAESTRO_APP_ROOT"]
sys.path.insert(0, app)
# Some optional diffusers/xformers imports ask about device capability. Keep
# this source-completeness check CPU-only without initializing a CUDA context.
import torch
torch.cuda.is_available = lambda: False
torch.cuda.get_device_capability = lambda device=None: (0, 0)
from services.memory_runtime import select_runtime, verify_runtime_import
selection = select_runtime(app, "wangp9905")
import mmgp
assert verify_runtime_import(selection) == "4.4.0"
modules = [item.name for item in pkgutil.walk_packages(mmgp.__path__, mmgp.__name__ + ".")]
for name in modules:
    importlib.import_module(name)

from mmgp import offload as offload_module
import mmgp.cuda_graphs as cuda_graphs_module
import mmgp.runtime as runtime_module

events = []
class Hook:
    def remove(self): events.append("hook")
class Residency:
    def close(self): events.append("residency")
class AutoPreload:
    def close(self): events.append("preload")
class Stager:
    def release(self): events.append("stager")
class Pool:
    def close(self, manager): events.append("pool")

manager = object.__new__(offload_module.offload)
manager.models = {}
manager.expert_hooks = [Hook()]
manager.expert_residency = {"test": Residency()}
manager.cuda_graphs = {"test": object()}
manager.auto_preload = AutoPreload()
manager.prefetch_executor = None
manager.inflight = {"job": object()}
manager._release_ring = lambda: events.append("ring")
manager.unload_all = lambda: events.append("unload")
manager.stager = Stager()
manager.block_items = {"block": object()}
manager.active_models = {"test"}
manager.default_stream = object()
manager.transfer_stream = object()
manager.parameters_ref = object()
manager.blocks_of_modules = {"test/block": object()}
with patch.object(offload_module, "_pool", Pool()), \
     patch.object(cuda_graphs_module, "release", lambda graphs: events.append("graphs")), \
     patch.object(runtime_module, "_unregister", lambda obj: events.append("unregister")), \
     patch.object(offload_module.gc, "collect", lambda: 0), \
     patch.object(offload_module.torch.cuda, "empty_cache", lambda: None):
    manager.release()
    manager.release()
assert {"unload", "hook", "residency", "graphs", "preload", "ring", "pool", "stager", "unregister"} <= set(events), events
assert manager.models is None
print("imported", len(modules), "MMGP modules; cleanup", ",".join(events))
'''
        environment = os.environ.copy()
        environment["MAESTRO_APP_ROOT"] = str(APP)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=APP,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("imported 19 MMGP modules", result.stdout)


if __name__ == "__main__":
    unittest.main()
