"""Temporary-filesystem tests for bounded ComfyUI model discovery."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from services import comfyui_models as comfy
except ModuleNotFoundError:
    from app.services import comfyui_models as comfy


def _isolated_user_env(home: Path) -> dict[str, str]:
    return {
        "USERPROFILE": str(home),
        "HOME": str(home),
        "APPDATA": str(home / "AppData" / "Roaming"),
        "LOCALAPPDATA": str(home / "AppData" / "Local"),
        "XDG_CONFIG_HOME": str(home / ".config"),
    }


def _write(path: Path, data: bytes = b"model") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _make_install(root: Path, *, portable: bool = False) -> Path:
    comfy_root = root / "ComfyUI" if portable else root
    _write(comfy_root / "main.py", b"# marker")
    _write(comfy_root / "folder_paths.py", b"# marker")
    (comfy_root / "models" / "checkpoints").mkdir(parents=True)
    return comfy_root


def _write_safetensors(path: Path, shape=(2, 3), dtype="F16") -> Path:
    header = {
        "model.weight": {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [0, 2],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _write(path, struct.pack("<Q", len(encoded)) + encoded + b"\0\0")


class TestComfyUIDiscovery(unittest.TestCase):
    def test_finds_sibling_install_and_portable_outer_root(self):
        with tempfile.TemporaryDirectory() as temp:
            pinokio = Path(temp)
            app_root = pinokio / "Maestro" / "app"
            app_root.mkdir(parents=True)
            sibling = pinokio / "ComfyUI"
            _make_install(sibling)
            portable = pinokio / "ComfyUI_windows_portable"
            _make_install(portable, portable=True)

            with mock.patch.dict(os.environ, _isolated_user_env(Path(temp) / "user")):
                found = comfy.discover_comfyui_installations(app_root)
            paths = {Path(item["path"]) for item in found}

            self.assertIn(sibling, paths)
            self.assertIn(portable, paths)
            portable_record = next(item for item in found if Path(item["path"]) == portable)
            self.assertEqual(portable_record["evidence"], "portable_comfyui_markers")
            self.assertTrue(portable_record["available"])

    def test_reads_relative_absolute_multiline_paths_and_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "ComfyUI"
            _make_install(root)
            shared = Path(temp) / "shared"
            relative_checkpoints = shared / "models" / "checkpoints"
            relative_loras = shared / "models" / "loras"
            relative_lora_alias = shared / "models" / "lora"
            unet = shared / "models" / "unet"
            clip = shared / "models" / "clip"
            absolute_vae = Path(temp) / "external" / "vae"
            env_models = Path(temp) / "environment-root" / "text_encoders"
            home_vae = Path(temp) / "fake-home" / "comfy" / "models" / "vae"
            for directory in (
                relative_checkpoints,
                relative_loras,
                relative_lora_alias,
                unet,
                clip,
                absolute_vae,
                env_models,
                home_vae,
            ):
                directory.mkdir(parents=True)

            config = root / "extra_model_paths.yaml"
            config.write_text(
                """
shared:
  base_path: ../shared
  checkpoints: models/checkpoints
  loras: |-
    models/loras
    models/lora
  unet: models/unet
  clip: models/clip
absolute:
  vae: %s
environment:
  base_path: "%%COMFY_TEST_BASE%%"
  text_encoders: text_encoders
home:
  base_path: ~/comfy
  vae: models/vae
""" % absolute_vae,
                encoding="utf-8",
            )

            env = _isolated_user_env(Path(temp) / "user")
            fake_home = Path(temp) / "fake-home"
            env.update(
                {
                    "COMFY_TEST_BASE": str(env_models.parent),
                    "HOME": str(fake_home),
                    "USERPROFILE": str(fake_home),
                }
            )
            with mock.patch.dict(os.environ, env):
                found = comfy.discover_comfyui_installations(root)
            install = next(item for item in found if Path(item["path"]) == root)
            paths = {(item["category"], Path(item["path"])) for item in install["model_paths"]}

            self.assertIn(("checkpoints", relative_checkpoints), paths)
            self.assertIn(("loras", relative_loras), paths)
            self.assertIn(("loras", relative_lora_alias), paths)
            self.assertIn(("diffusion_models", unet), paths)
            self.assertIn(("text_encoders", clip), paths)
            self.assertIn(("vae", absolute_vae), paths)
            self.assertIn(("vae", home_vae), paths)
            self.assertIn(("text_encoders", env_models), paths)

    def test_reads_desktop_config_and_deduplicates_model_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            pinokio = Path(temp) / "pinokio"
            app_root = pinokio / "Maestro" / "app"
            app_root.mkdir(parents=True)
            unrelated = pinokio / "ComfyUI"
            _make_install(unrelated)
            user = Path(temp) / "user"
            local_appdata = user / "AppData" / "Local"
            desktop_install = local_appdata / "Programs" / "ComfyUI"
            _make_install(desktop_install)
            desktop_models = Path(temp) / "desktop-models" / "loras"
            desktop_checkpoints = Path(temp) / "desktop-models" / "models" / "checkpoints"
            desktop_models.mkdir(parents=True)
            desktop_checkpoints.mkdir(parents=True)
            appdata = user / "AppData" / "Roaming"
            config = appdata / "ComfyUI" / "extra_models_config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "desktop:\n  base_path: ../../../../desktop-models\n  loras: loras\n",
                encoding="utf-8",
            )
            # The installation-local and Desktop files may both describe the
            # same path. Category/path pairs should appear once.
            (desktop_install / "extra_model_paths.yaml").write_text(
                f"desktop:\n  loras: {desktop_models}\n", encoding="utf-8"
            )

            with mock.patch.dict(os.environ, _isolated_user_env(user)):
                found = comfy.discover_comfyui_installations(app_root)

            install = next(item for item in found if Path(item["path"]) == desktop_install)
            matches = [
                item for item in install["model_paths"]
                if item["category"] == "loras" and Path(item["path"]) == desktop_models
            ]
            self.assertEqual(len(matches), 1)
            self.assertIn(
                {"category": "checkpoints", "path": str(desktop_checkpoints)},
                install["model_paths"],
            )
            unrelated_record = next(item for item in found if Path(item["path"]) == unrelated)
            self.assertNotIn(
                str(desktop_models),
                {item["path"] for item in unrelated_record["model_paths"]},
            )

    def test_malformed_yaml_and_missing_configured_roots_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "ComfyUI"
            _make_install(root)
            (root / "extra_model_paths.yaml").write_text(
                "checkpoints: [unterminated", encoding="utf-8"
            )
            (root / "extra_models_config.yaml").write_text(
                "extra:\n  base_path: missing\n  vae: absent\n", encoding="utf-8"
            )

            with mock.patch.dict(os.environ, _isolated_user_env(Path(temp) / "user")):
                found = comfy.discover_comfyui_installations(root)
            install = next(item for item in found if Path(item["path"]) == root)

            self.assertEqual(
                install["model_paths"],
                [{"category": "checkpoints", "path": str(root / "models" / "checkpoints")}],
            )
            missing = comfy.scan_comfyui_models([str(Path(temp) / "missing")])
            self.assertEqual(missing["files"], [])
            self.assertFalse(missing["roots"][0]["available"])

    def test_explicit_model_root_is_reported_without_claiming_an_install(self):
        with tempfile.TemporaryDirectory() as temp:
            model_root = Path(temp) / "models" / "checkpoints"
            model_root.mkdir(parents=True)
            with mock.patch.dict(os.environ, _isolated_user_env(Path(temp) / "user")):
                found = comfy.discover_comfyui_installations(Path(temp) / "Maestro", [str(model_root)])
            configured = next(item for item in found if item["source"] == "configured_model_root")
            self.assertEqual(Path(configured["path"]), model_root)
            self.assertEqual(configured["model_paths"], [{"category": "checkpoints", "path": str(model_root)}])


class TestComfyUIScanning(unittest.TestCase):
    def test_scans_alias_roots_and_deduplicates_overlapping_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            models = Path(temp) / "models"
            checkpoints = models / "checkpoints"
            unet = models / "unet"
            checkpoint_file = _write(checkpoints / "one.safetensors")
            unet_file = _write(unet / "two.gguf")

            result = comfy.scan_comfyui_models([str(models), str(checkpoints), str(models)])

            by_name = {item["filename"]: item for item in result["files"]}
            self.assertEqual(set(by_name), {"one.safetensors", "two.gguf"})
            self.assertEqual(by_name[checkpoint_file.name]["category"], "checkpoints")
            self.assertEqual(by_name[unet_file.name]["category"], "diffusion_models")
            self.assertFalse(result["truncated"])

    def test_known_hash_proves_match_but_filename_alone_does_not(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoints"
            data = b"verified bytes"
            candidate = _write(root / "exact.safetensors", data)
            _write(root / "familiar.safetensors", b"different bytes")
            known = [{
                "id": "asset-1",
                "filename": "exact.safetensors",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "category": "checkpoints",
            }, {"filename": "familiar.safetensors", "category": "checkpoints"}]

            result = comfy.scan_comfyui_models([str(root)], known)
            by_name = {item["filename"]: item for item in result["files"]}

            self.assertEqual(by_name[candidate.name]["compatibility"], "native")
            self.assertEqual(by_name[candidate.name]["known_asset_id"], "asset-1")
            self.assertEqual(by_name["familiar.safetensors"]["compatibility"], "needs_review")

    def test_tensor_layout_evidence_is_header_only_and_mismatch_needs_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "loras"
            matching = _write_safetensors(root / "layout.safetensors")
            mismatch = _write_safetensors(root / "mismatch.safetensors", shape=(2, 4))
            known = [
                {
                    "filename": matching.name,
                    "category": "loras",
                    "tensor_layout": {"model.weight": {"shape": [2, 3], "dtype": "F16"}},
                },
                {
                    "filename": mismatch.name,
                    "category": "loras",
                    "tensor_layout": {"model.weight": {"shape": [2, 3], "dtype": "F16"}},
                },
            ]

            result = comfy.scan_comfyui_models([str(root)], known)
            by_name = {item["filename"]: item for item in result["files"]}

            self.assertEqual(by_name[matching.name]["compatibility"], "native")
            self.assertEqual(by_name[mismatch.name]["compatibility"], "needs_review")

    def test_ambiguous_same_filename_expectations_never_claim_native(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "models"
            _write(root / "checkpoints" / "same.safetensors", b"same")
            known = [
                {"filename": "same.safetensors", "size": 4, "sha256": "a" * 64},
                {"filename": "same.safetensors", "size": 4, "sha256": "b" * 64},
            ]

            result = comfy.scan_comfyui_models([str(root)], known)

            self.assertEqual(len(result["files"]), 1)
            self.assertEqual(result["files"][0]["compatibility"], "needs_review")
            self.assertIn("Multiple known assets", result["files"][0]["reason"])

    def test_scan_bounds_depth_and_file_count_and_reports_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "models"
            _write(root / "checkpoints" / "a.safetensors")
            _write(root / "checkpoints" / "b.safetensors")
            _write(root / "checkpoints" / "c.safetensors")
            _write(root / "checkpoints" / "nested" / "deeper.safetensors")

            with mock.patch.object(comfy, "MAX_SCAN_FILES", 2):
                bounded = comfy.scan_comfyui_models([str(root)])
            self.assertEqual(len(bounded["files"]), 2)
            self.assertTrue(bounded["truncated"])

            with mock.patch.object(comfy, "MAX_SCAN_DEPTH", 0):
                depth_limited = comfy.scan_comfyui_models([str(root)])
            self.assertTrue(depth_limited["truncated"])

    def test_symlink_cycle_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoints"
            _write(root / "one.safetensors")
            try:
                (root / "loop").symlink_to(root, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Directory symlinks are unavailable in this environment")

            result = comfy.scan_comfyui_models([str(root)])

            self.assertEqual([item["filename"] for item in result["files"]], ["one.safetensors"])
            self.assertFalse(result["truncated"])

    def test_unreadable_paths_and_individual_scan_errors_are_best_effort(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoints"
            root.mkdir()
            with mock.patch.object(comfy.os, "scandir", side_effect=PermissionError("denied")):
                discovered = comfy.discover_comfyui_installations(root)
                scanned = comfy.scan_comfyui_models([str(root)])

            self.assertEqual(discovered, [])
            self.assertFalse(scanned["roots"][0]["available"])
            self.assertEqual(scanned["files"], [])


if __name__ == "__main__":
    unittest.main()
