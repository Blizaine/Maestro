"""Regression tests for MiniMax H3 x2 VAE download and lookup paths."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from models.minimax_h3 import vae_upsampler  # noqa: E402


_HANDLER_PATH = APP / "models" / "minimax_h3" / "minimax_h3_handler.py"


def _load_handler_module():
    spec = importlib.util.spec_from_file_location("h3_x2_download_handler_test", _HANDLER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load MiniMax H3 handler for regression tests")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _process_files_def(descriptor, target_root, checkpoint_roots):
    """Exercise the production downloader's real target/source-folder logic."""
    tree = ast.parse((APP / "wgp.py").read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "process_files_def"
    )
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        directory = kwargs["local_dir"]
        if kwargs.get("subfolder"):
            directory = os.path.join(directory, kwargs["subfolder"])
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, kwargs["filename"])
        Path(path).write_bytes(b"downloaded test asset")
        return path

    fake_locator = types.SimpleNamespace(
        get_smart_download_root=lambda _target: str(target_root),
        get_checkpoints_paths=lambda: [str(path) for path in checkpoint_roots],
    )
    namespace = {
        "os": os,
        "fl": fake_locator,
        "hf_download_with_public_fallback": fake_download,
        "snapshot_download": mock.Mock(),
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(APP / "wgp.py"), "exec"), namespace)
    namespace["process_files_def"](**descriptor)
    return calls


class H3X2VaeDownloadPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler = _load_handler_module()
        cls.model_def = cls.handler.family_handler.query_model_def("minimax_h3", {})

    @classmethod
    def _x2_descriptors(cls, roots):
        with mock.patch.object(cls.handler.fl, "get_checkpoints_paths", return_value=[str(root) for root in roots]):
            downloads = cls.handler.family_handler.query_model_files(
                [], "minimax_h3", cls.model_def, VAE_upsampling=vae_upsampler.X2_VAE_VALUE
            )
        return [item for item in downloads if item.get("repoId") == vae_upsampler.X2_VAE_REPO]

    @staticmethod
    def _full_descriptor():
        return {
            "repoId": vae_upsampler.X2_VAE_REPO,
            "revision": vae_upsampler.X2_VAE_REVISION,
            "sourceFolderList": [vae_upsampler.X2_VAE_REPO_SUBFOLDER],
            "targetFolderList": [os.path.join("minimax_h3", "vae")],
            "fileList": [[
                vae_upsampler.X2_VAE_FILE,
                vae_upsampler.X2_VAE_LICENSE_FILE,
                vae_upsampler.X2_VAE_NOTICE_FILE,
            ]],
        }

    def test_first_download_uses_hugging_face_subfolder_and_resolver_finds_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            descriptors = self._x2_descriptors([root])
            self.assertEqual(len(descriptors), 1)
            descriptor = descriptors[0]
            self.assertEqual(descriptor["revision"], vae_upsampler.X2_VAE_REVISION)
            self.assertEqual(descriptor["sourceFolderList"], [vae_upsampler.X2_VAE_REPO_SUBFOLDER])
            self.assertEqual(descriptor["targetFolderList"], [os.path.join("minimax_h3", "vae")])
            self.assertEqual(descriptor["fileList"], [[
                vae_upsampler.X2_VAE_FILE,
                vae_upsampler.X2_VAE_LICENSE_FILE,
                vae_upsampler.X2_VAE_NOTICE_FILE,
            ]])

            calls = _process_files_def(descriptor, root, [root])
            expected = root / "minimax_h3" / "vae" / vae_upsampler.X2_VAE_REPO_SUBFOLDER / vae_upsampler.X2_VAE_FILE
            self.assertEqual(len(calls), 3)
            self.assertTrue(expected.is_file())
            with mock.patch.object(vae_upsampler, "verify_x2_vae_checkpoint") as verify:
                resolved = vae_upsampler.resolve_x2_vae_checkpoint("minimax_h3", [root])
            self.assertEqual(Path(resolved), expected)
            verify.assert_called_once_with(str(expected))

    def test_flat_canonical_weight_is_reused_while_missing_sidecars_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = root / "minimax_h3" / "vae" / vae_upsampler.X2_VAE_FILE
            canonical.parent.mkdir(parents=True)
            canonical.write_bytes(b"existing canonical weight")

            descriptors = self._x2_descriptors([root])
            self.assertEqual(len(descriptors), 1)
            descriptor = descriptors[0]
            self.assertEqual(descriptor["fileList"], [[
                vae_upsampler.X2_VAE_LICENSE_FILE,
                vae_upsampler.X2_VAE_NOTICE_FILE,
            ]])
            calls = _process_files_def(descriptor, root, [root])
            self.assertEqual(
                {call["filename"] for call in calls},
                {vae_upsampler.X2_VAE_LICENSE_FILE, vae_upsampler.X2_VAE_NOTICE_FILE},
            )
            self.assertTrue(canonical.is_file())
            self.assertEqual(canonical.read_bytes(), b"existing canonical weight")
            with mock.patch.object(vae_upsampler, "verify_x2_vae_checkpoint"):
                self.assertEqual(Path(vae_upsampler.resolve_x2_vae_checkpoint("minimax_h3", [root])), canonical)

    def test_complete_nested_shared_root_is_reused_without_any_download(self):
        with tempfile.TemporaryDirectory() as temp:
            primary = Path(temp) / "primary"
            shared = Path(temp) / "shared"
            nested = shared / "minimax_h3" / "vae" / vae_upsampler.X2_VAE_REPO_SUBFOLDER
            nested.mkdir(parents=True)
            for name in (
                vae_upsampler.X2_VAE_FILE,
                vae_upsampler.X2_VAE_LICENSE_FILE,
                vae_upsampler.X2_VAE_NOTICE_FILE,
            ):
                (nested / name).write_bytes(b"shared test asset")

            self.assertEqual(self._x2_descriptors([primary, shared]), [])
            calls = _process_files_def(self._full_descriptor(), primary, [primary, shared])
            self.assertEqual(calls, [])
            with mock.patch.object(vae_upsampler, "verify_x2_vae_checkpoint"):
                resolved = vae_upsampler.resolve_x2_vae_checkpoint("minimax_h3", [primary, shared])
            self.assertEqual(Path(resolved), nested / vae_upsampler.X2_VAE_FILE)
            self.assertEqual(sorted(path.name for path in nested.iterdir()), sorted([
                vae_upsampler.X2_VAE_FILE,
                vae_upsampler.X2_VAE_LICENSE_FILE,
                vae_upsampler.X2_VAE_NOTICE_FILE,
            ]))

    def test_invalid_preferred_copy_falls_through_to_valid_shared_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            primary = Path(temp) / "primary"
            shared = Path(temp) / "shared"
            preferred = primary / "minimax_h3" / "vae" / vae_upsampler.X2_VAE_FILE
            fallback = shared / "minimax_h3" / "vae" / vae_upsampler.X2_VAE_REPO_SUBFOLDER / vae_upsampler.X2_VAE_FILE
            preferred.parent.mkdir(parents=True)
            fallback.parent.mkdir(parents=True)
            preferred.write_bytes(b"bad preferred copy")
            fallback.write_bytes(b"valid shared copy")

            def verify(path):
                if Path(path) == preferred:
                    raise ValueError("bad pinned hash")

            with mock.patch.object(vae_upsampler, "verify_x2_vae_checkpoint", side_effect=verify):
                resolved = vae_upsampler.resolve_x2_vae_checkpoint("minimax_h3", [primary, shared])
            self.assertEqual(Path(resolved), fallback)
            self.assertTrue(preferred.is_file())
            self.assertTrue(fallback.is_file())


if __name__ == "__main__":
    unittest.main()