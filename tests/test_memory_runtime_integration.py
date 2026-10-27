"""CPU-only integration checks for the saved experimental runtime boundary."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
sys.path.insert(0, str(APP))

from services.memory_runtime import (  # noqa: E402
    EXPERIMENTAL_PROFILE0_RUNTIME,
    LEGACY_RUNTIME,
    select_runtime,
    validate_system_memory_updates,
)


class MemoryRuntimeSystemBoundaryTests(unittest.TestCase):
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

    @staticmethod
    def make_app(root: Path) -> Path:
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

    def test_invalid_mixed_system_update_is_rejected_without_mutating_saved_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.make_app(Path(temporary))
            active = select_runtime(app, LEGACY_RUNTIME)
            allocator_dir = app / "mmgp_backends" / "v43" / "mmgp" / "allocator"
            allocator_dir.mkdir(parents=True)
            (allocator_dir / "vmm_alloc_win_amd64.dll").write_bytes(b"fixture-only")
            cfg = {
                "memory_runtime": LEGACY_RUNTIME,
                "vram_allocator": "default",
                "ram_allocator": "default",
                "video_profile": 3,
                "image_profile": 4,
                "audio_profile": 5,
                "read_ahead": False,
                "smart_memory_pinning": False,
                "video_preload_mode": "manual",
                "video_preload_in_VRAM": 3200,
            }
            body = {
                "memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME,
                "video_profile": 1,
                "read_ahead": True,
            }
            cfg_before, body_before = dict(cfg), dict(body)
            with self.assertRaisesRegex(ValueError, "removed read-ahead"):
                validate_system_memory_updates(body, cfg, active)
            self.assertEqual(cfg, cfg_before)
            self.assertEqual(body, body_before)

    def test_runtime_selection_does_not_rewrite_manual_profile_or_preload_choices(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.make_app(Path(temporary))
            active = select_runtime(app, LEGACY_RUNTIME)
            allocator_dir = app / "mmgp_backends" / "v43" / "mmgp" / "allocator"
            allocator_dir.mkdir(parents=True)
            (allocator_dir / "vmm_alloc_win_amd64.dll").write_bytes(b"fixture-only")
            cfg = {
                "memory_runtime": LEGACY_RUNTIME,
                "vram_allocator": "default",
                "ram_allocator": "default",
                "video_profile": 3.5,
                "image_profile": 4.5,
                "audio_profile": 5,
                "smart_memory_pinning": False,
                "video_preload_mode": "manual",
                "video_preload_in_VRAM": 1024,
                "image_preload_mode": "dynamic",
            }
            before = dict(cfg)
            memory_updates, runtime_updates = validate_system_memory_updates(
                {"memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME}, cfg, active
            )
            self.assertEqual(memory_updates, {})
            self.assertEqual(runtime_updates, {"memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME})
            self.assertEqual(cfg, before)

    def test_backend_switch_fails_closed_when_pinned_vmm_asset_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            active = select_runtime(self.make_app(Path(temporary)), LEGACY_RUNTIME)
            cfg = {"memory_runtime": LEGACY_RUNTIME, "video_profile": 2, "vram_allocator": "default"}
            before = dict(cfg)
            with self.assertRaisesRegex(ValueError, "cannot be selected or saved"):
                validate_system_memory_updates(
                    {"memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME}, cfg, active
                )
            self.assertEqual(cfg, before)

    def test_missing_asset_does_not_block_explicit_recovery_to_legacy_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            active = select_runtime(self.make_app(Path(temporary)), LEGACY_RUNTIME)
            cfg = {
                "memory_runtime": EXPERIMENTAL_PROFILE0_RUNTIME,
                "video_profile": 0,
                "image_profile": 4,
                "audio_profile": 5,
                "vram_allocator": "vmm",
                "ram_allocator": "default",
                "read_ahead": False,
            }
            profile_update = validate_system_memory_updates(
                {"video_profile": 1}, cfg, active
            )
            self.assertEqual(profile_update[1], {"video_profile": 1})
            recovered = {**cfg, "video_profile": 1}
            self.assertEqual(
                validate_system_memory_updates(
                    {"memory_runtime": LEGACY_RUNTIME}, recovered, active
                )[1],
                {"memory_runtime": LEGACY_RUNTIME},
            )

    def test_api_preflights_runtime_and_linked_folders_before_mutation(self):
        source = (APP / "launch.py").read_text(encoding="utf-8")
        route_start = source.index("async def update_system_config(request: Request):")
        route = source[route_start:source.index("@api.post(\"/api/v1/notification-sound/test\")", route_start)]
        self.assertLess(route.index("validate_system_memory_updates("), route.index("updated = {}"))
        self.assertLess(route.index("_validate_linked_model_folders(body[\"model_folders\"])", route.index("# Preflight")), route.index("updated = {}"))
        self.assertIn("host_notification_sound_enabled must be true or false", route)
        self.assertIn("No valid fields to update", route)

    def test_ui_exposes_runtime_restart_and_profile0_capability_gates(self):
        memory_panel = (ROOT / "ui/src/components/SettingsDrawer/MemorySettingsPanel.tsx").read_text(encoding="utf-8")
        system_panel = (ROOT / "ui/src/components/SettingsDrawer/SystemSettingsPanel.tsx").read_text(encoding="utf-8")
        self.assertIn('data-testid="memory-runtime-select"', memory_panel)
        self.assertIn("disabled={!config.experimental_runtime_selectable}", memory_panel)
        self.assertIn("experimental_runtime_capability_reason", memory_panel)
        self.assertIn("Restart Maestro to apply the saved memory runtime", memory_panel)
        self.assertIn("Fish Audio is unsupported", memory_panel)
        self.assertIn("Profile 0 uses a shared pin pool", memory_panel)
        self.assertIn("disabled={runtimeExperimental && !(config.read_ahead ?? false)}", memory_panel)
        self.assertIn("profile0Allowed = systemConfig.saved_runtime === 'wangp9905'", system_panel)
        self.assertIn("systemConfig.vram_allocator === 'vmm'", system_panel)
        self.assertIn("systemConfig.profile0_native_available === true", system_panel)
        self.assertIn("disabled: !profile0Allowed", system_panel)
        self.assertIn("changes model placement for Profiles 1 and 3", system_panel)


if __name__ == "__main__":
    unittest.main()
