"""CPU-only regressions for the retired MiniMax H3 VDN download pin."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from models.minimax_h3 import vdn_assets


def _url(revision: str, asset_path: str, repo_id: str = vdn_assets.VDN_REPO_ID) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{asset_path}"


def _load_wgp_functions(*names: str) -> dict:
    source = _APP / "wgp.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    wanted = set(names)
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in functions} != wanted:
        raise AssertionError(f"Could not extract requested functions: {wanted}")
    namespace = {"json": json, "os": os, "re": re, "shutil": shutil}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class H3VdnDownloadTests(unittest.TestCase):
    def test_manifest_records_the_four_verified_files(self):
        self.assertEqual(
            vdn_assets.VDN_ASSET_SPECS,
            {
                "modules/minimax_h3_vdn_module_bf16.safetensors": {
                    "size": 4_279_428_112,
                    "sha256": "dec6981c7874f5b3bc92d1a02e256b673a3b3499dc1a124714bb3b19da602855",
                },
                "modules/minimax_h3_vdn_module_int8_convrot.safetensors": {
                    "size": 2_166_148_396,
                    "sha256": "dc22ad23ebee2ed893e79b9289ae6608d6eaa55f7a69e4c9bb8c19816e7b0c36",
                },
                "loras/minimax_h3_vdn_default_lora_bf16.safetensors": {
                    "size": 334_026_912,
                    "sha256": "58558fef506f88bb41649242de9b9b3a365da806b51b2e96afbbe1625222058a",
                },
                "loras/minimax_h3_vdn_turbo_8step_lora_bf16.safetensors": {
                    "size": 851_452_696,
                    "sha256": "24fc93c82fe84dc45d0627f4e72c637bc387d282ba18f60ed3b7f8c81089392c",
                },
            },
        )

    def test_repair_is_exact_and_preserves_query_and_fragment(self):
        for asset_path in vdn_assets.VDN_ASSET_SPECS:
            with self.subTest(asset_path=asset_path):
                old_url = _url(vdn_assets.VDN_LEGACY_REVISION, asset_path)
                pinned_url = _url(vdn_assets.VDN_REVISION, asset_path)
                self.assertEqual(vdn_assets.canonicalize_vdn_asset_url(old_url), pinned_url)
                self.assertEqual(vdn_assets.canonicalize_vdn_asset_url(pinned_url), pinned_url)

        asset_path = "loras/minimax_h3_vdn_default_lora_bf16.safetensors"
        old_url = _url(vdn_assets.VDN_LEGACY_REVISION, asset_path)
        self.assertEqual(
            vdn_assets.canonicalize_vdn_asset_url(old_url + "?download=true&source=cache#weights"),
            _url(vdn_assets.VDN_REVISION, asset_path) + "?download=true&source=cache#weights",
        )

        unchanged = (
            _url(vdn_assets.VDN_LEGACY_REVISION, "README.md"),
            _url("main", asset_path),
            _url(vdn_assets.VDN_LEGACY_REVISION, asset_path, "AnotherOwner/MiniMax-H3"),
            "https://example.com/file.safetensors",
        )
        for url in unchanged:
            with self.subTest(url=url):
                self.assertEqual(vdn_assets.canonicalize_vdn_asset_url(url), url)

    def test_all_four_assets_download_with_the_repaired_pin_and_original_basenames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = []

            def fake_hf_download(**kwargs):
                calls.append(kwargs)
                path = Path(kwargs["local_dir"]) / kwargs["subfolder"] / kwargs["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"checkpoint fixture")

            namespace = _load_wgp_functions("download_file")
            namespace.update({
                "fl": SimpleNamespace(get_download_location=lambda: str(root / "downloads")),
                "hf_download_with_public_fallback": fake_hf_download,
            })
            for asset_path in vdn_assets.VDN_ASSET_SPECS:
                with self.subTest(asset_path=asset_path):
                    target = root / "installed" / Path(asset_path).name
                    namespace["download_file"](
                        _url(vdn_assets.VDN_LEGACY_REVISION, asset_path), str(target),
                    )
                    self.assertEqual(target.read_bytes(), b"checkpoint fixture")
                    self.assertEqual(calls[-1]["revision"], vdn_assets.VDN_REVISION)
                    self.assertEqual(calls[-1]["repo_id"], vdn_assets.VDN_REPO_ID)
                    self.assertEqual(calls[-1]["filename"], target.name)
                    self.assertEqual(calls[-1]["subfolder"], asset_path.split("/")[0])
            self.assertEqual(len(calls), 4)

    def test_saved_legacy_preset_cache_downloads_through_wgp_hook(self):
        asset_path = "loras/minimax_h3_vdn_turbo_8step_lora_bf16.safetensors"
        old_url = _url(vdn_assets.VDN_LEGACY_REVISION, asset_path)
        filename = Path(asset_path).name

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lora_dir = root / "loras"
            lora_dir.mkdir()
            download_root = root / "downloads"
            calls = []

            def fake_hf_download(**kwargs):
                calls.append(kwargs)
                destination = Path(kwargs["local_dir"])
                if kwargs.get("subfolder"):
                    destination /= kwargs["subfolder"]
                destination.mkdir(parents=True, exist_ok=True)
                (destination / kwargs["filename"]).write_bytes(b"test VDN LoRA")

            namespace = _load_wgp_functions(
                "download_file", "check_loras_exist", "update_loras_url_cache"
            )
            namespace.update({
                "fl": SimpleNamespace(get_download_location=lambda: str(download_root)),
                "hf_download_with_public_fallback": fake_hf_download,
                "get_lora_dir": lambda _model_type: str(lora_dir),
                "resolve_lora_path": lambda _model_type, value: str(
                    lora_dir / os.path.basename(str(value))
                ),
                "loras_url_cache": None,
            })
            namespace["_ensure_loras_url_cache"] = lambda: namespace[
                "update_loras_url_cache"
            ]("", [])

            previous_cwd = os.getcwd()
            try:
                os.chdir(root)
                preset_path = root / "Saved VDN Turbo.json"
                preset_path.write_text(
                    json.dumps({"activated_loras": [old_url]}), encoding="utf-8"
                )
                saved_preset = json.loads(preset_path.read_text(encoding="utf-8"))

                # Loading an older saved preset records its stale source URL in
                # the same persistent LoRA URL cache used by settings imports.
                selected_loras = namespace["update_loras_url_cache"](
                    str(lora_dir), saved_preset["activated_loras"]
                )
                self.assertEqual(selected_loras, [filename])
                cache_path = root / "loras_url_cache.json"
                self.assertEqual(
                    json.loads(cache_path.read_text(encoding="utf-8")),
                    {str(lora_dir / filename): old_url},
                )

                # Reload the persisted cache and let the real extracted
                # check_loras_exist -> download_file flow resolve the old URL.
                namespace["loras_url_cache"] = None
                error = namespace["check_loras_exist"](
                    "minimax_h3_vdn", selected_loras, download=True
                )
                target = lora_dir / filename
                self.assertEqual(error, "")
                self.assertEqual(target.read_bytes(), b"test VDN LoRA")
                self.assertEqual(target.name, filename)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["repo_id"], vdn_assets.VDN_REPO_ID)
                self.assertEqual(calls[0]["revision"], vdn_assets.VDN_REVISION)
                self.assertEqual(calls[0]["subfolder"], "loras")
                self.assertEqual(calls[0]["filename"], filename)

                # A present cached basename keeps its path and bytes without a
                # second request or a large-file rehash.
                target.write_bytes(b"existing cached copy")
                self.assertEqual(
                    namespace["check_loras_exist"](
                        "minimax_h3_vdn", selected_loras, download=True
                    ),
                    "",
                )
                self.assertEqual(target.read_bytes(), b"existing cached copy")
                self.assertEqual(len(calls), 1)
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
