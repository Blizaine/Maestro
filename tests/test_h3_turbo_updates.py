"""Regression tests for curated MiniMax H3 Turbo updates."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
_MANAGED_ASSETS_PATH = _APP / "services" / "managed_assets.py"
_MONITOR_PATH = _ROOT / "scripts" / "check_h3_turbo_upstream.py"
_MANIFEST_PATH = _APP / "models" / "minimax_h3" / "turbo_presets.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManagedTurboAssetTests(unittest.TestCase):
    def test_manifest_promotes_v4_default_and_keeps_legacy_rollback(self):
        manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        presets = {preset["id"]: preset for preset in manifest["presets"]}
        self.assertEqual(manifest["default_preset_id"], "v4-step600-ema")
        self.assertEqual(presets["v1-ckpt500"]["status"], "legacy")
        current = presets["v4-step600-ema"]
        self.assertEqual(current["status"], "validated")
        self.assertEqual(current["steps"], 6)
        self.assertEqual(current["weight"], 1.0)
        self.assertEqual(current["size"], 779_849_816)
        self.assertEqual(
            current["sha256"],
            "5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3",
        )

    def test_managed_asset_receipt_revalidates_changed_manifest_or_file(self):
        managed_assets = _load_module("maestro_managed_assets_test", _MANAGED_ASSETS_PATH)
        payload = b"maestro-managed-turbo"
        digest = hashlib.sha256(payload).hexdigest()
        spec = {
            "repo_id": "example/turbo",
            "revision": "revision-a",
            "remote_path": "turbo.safetensors",
            "sha256": digest,
            "size": len(payload),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            asset = Path(temp_dir) / "turbo.safetensors"
            asset.write_bytes(payload)
            self.assertTrue(managed_assets.managed_asset_matches(str(asset), spec))
            receipt = Path(managed_assets.managed_asset_receipt_path(str(asset)))
            self.assertTrue(receipt.is_file())

            revised_spec = {**spec, "revision": "revision-b"}
            self.assertTrue(
                managed_assets.managed_asset_matches(str(asset), revised_spec)
            )
            revised_receipt = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(revised_receipt["revision"], "revision-b")

            asset.write_bytes(b"x" * len(payload))
            self.assertFalse(
                managed_assets.managed_asset_matches(str(asset), revised_spec)
            )


class TurboUpstreamMonitorTests(unittest.TestCase):
    def test_monitor_detects_only_unreviewed_revision(self):
        monitor = _load_module("maestro_h3_turbo_monitor_test", _MONITOR_PATH)
        manifest = monitor.load_manifest(_MANIFEST_PATH)
        observed = manifest["upstream_watch"]["observed_main_revision"]
        payload = {
            "sha": observed,
            "lastModified": "2026-08-08T20:07:30Z",
            "siblings": [
                {"rfilename": "README.md"},
                {"rfilename": "candidate.safetensors"},
            ],
        }
        tree_payload = [
            {"path": "README.md", "type": "file", "size": 123},
            {
                "path": "candidate.safetensors",
                "type": "file",
                "size": 456,
                "lfs": {"oid": "content-sha256", "size": 456},
                "xetHash": "storage-object-hash",
            },
        ]

        def opener(request, timeout):
            self.assertEqual(timeout, 30)
            response_payload = (
                tree_payload if "/tree/" in request.full_url else payload
            )
            return io.BytesIO(json.dumps(response_payload).encode("utf-8"))

        upstream = monitor.fetch_upstream_state(manifest["repo_id"], opener=opener)
        result = monitor.compare_manifest_to_upstream(manifest, upstream)
        self.assertFalse(result["changed"])
        self.assertEqual(result["safetensors"], ["candidate.safetensors"])
        self.assertEqual(
            result["safetensor_metadata"],
            [
                {
                    "filename": "candidate.safetensors",
                    "size": 456,
                    "sha256": "content-sha256",
                    "xet_hash": "storage-object-hash",
                }
            ],
        )

        upstream["latest_revision"] = "new-revision"
        result = monitor.compare_manifest_to_upstream(manifest, upstream)
        self.assertTrue(result["changed"])
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "github-output.txt"
            monitor.write_github_outputs(output, result)
            values = output.read_text(encoding="utf-8")
        self.assertIn("changed=true", values)
        self.assertIn("latest_revision=new-revision", values)
        self.assertIn('"sha256":"content-sha256"', values)


if __name__ == "__main__":
    unittest.main()
