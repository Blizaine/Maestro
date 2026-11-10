"""Model-free RelayVSR routing, scale, and checkpoint-integrity contracts."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from postprocessing.relayvsr import decoder, wgp_bridge
from services.editor_projects import create_editor_project, normalize_editor_project
from services.media_info import processing_record
from services.media_processing import validate_methods

KEYFRAME_SKIP_REASON = "RelayVSR keyframe model tests require optional Wan pose dependency smplfitter"
try:
    from postprocessing.relayvsr import keyframes
except ModuleNotFoundError as error:
    if (error.name or "").partition(".")[0] != "smplfitter":
        raise
    keyframes = None


class RelayVSRIntegrationTests(unittest.TestCase):
    @unittest.skipIf(keyframes is None, KEYFRAME_SKIP_REASON)
    def test_keyframe_block_uses_group_aware_attention_hook(self):
        class FakeSelfAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Linear(32, 32, bias=False)
                self.k = torch.nn.Linear(32, 32, bias=False)
                self.v = torch.nn.Linear(32, 32, bias=False)
                self.o = torch.nn.Linear(32, 32, bias=False)
                self.num_heads = 1
                self.head_dim = 32

            @staticmethod
            def _norm_rope_(query, key, group, freqs):
                return None

        with patch.object(keyframes, "runtime", None), patch.object(
            keyframes, "pay_attention", side_effect=lambda qkv: qkv[0]
        ):
            cache = keyframes.KeyframeCache(
                [object()], tokens=2, height=1, width=2, heads=1, head_dim=32,
                device=torch.device("cpu"), in_ram=True,
            )
            block = keyframes.RelayWanBlock.__new__(keyframes.RelayWanBlock)
            torch.nn.Module.__init__(block)
            block.modulation = SimpleNamespace(weight=torch.zeros((1, 6, 32)))
            block.self_attn = FakeSelfAttention()
            block.norm1 = torch.nn.Identity()
            block.norm2 = torch.nn.Identity()
            block.norm3 = torch.nn.Identity()
            block.cross_attn = lambda states, _context, _cache: torch.zeros_like(states[0])
            block.ffn = torch.nn.Identity()

            output = block.forward(
                torch.zeros((1, 2, 32)), torch.zeros((1, 6, 32)), None, None,
                cache.attention_fn(0), lambda: None,
            )

            self.assertEqual(tuple(output.shape), (1, 2, 32))
            self.assertEqual(cache.layers[0][0].device, torch.device("cpu"))
            cache.release()

    def test_decoder_value_norm_uses_group_aware_attention_hook(self):
        attention = decoder.Attention()
        rope = decoder.rope_table(1, 2, torch.device("cpu"))
        groups = []

        def consume(qkv, group):
            groups.append(group)
            return qkv[0]

        output = attention.attend(
            [torch.randn((1, 2, decoder.DIM))], rope, consume, condition=False
        )

        self.assertEqual(tuple(output.shape), (1, 2, decoder.DIM))
        self.assertEqual(len(groups), 1)
        self.assertEqual(tuple(groups[0].kv), tuple(range(decoder.KV_HEADS)))

    @unittest.skipIf(keyframes is None, KEYFRAME_SKIP_REASON)
    def test_legacy_mmgp_cache_uses_bounded_host_int8_state(self):
        with patch.object(keyframes, "runtime", None), patch.object(
            keyframes, "pay_attention", side_effect=lambda qkv: qkv[0]
        ):
            cache = keyframes.KeyframeCache(
                [object()], tokens=2, height=1, width=2, heads=1, head_dim=32,
                device=torch.device("cpu"), in_ram=True,
            )
            self.assertIsNone(cache.ring)
            self.assertEqual(cache.cache_device, torch.device("cpu"))
            self.assertEqual(cache.layers[0][0].dtype, torch.int8)
            self.assertEqual(cache.layers[0][0].device, torch.device("cpu"))

            qkv = [torch.ones((1, 2, 1, 32)) for _ in range(3)]
            result = cache.attention_fn(0)(qkv, SimpleNamespace(kv=range(1)))
            self.assertEqual(tuple(result.shape), (1, 2, 1, 32))
            self.assertEqual(cache.layers[0][0].shape, (8, 1, 32))
            self.assertEqual(cache.layers[0][0].device, torch.device("cpu"))
            cache.commit(0)
            cache.release()

    def test_supported_method_ids_keep_scale_and_editor_contract(self):
        for scale in wgp_bridge.SUPPORTED_SCALES:
            method = wgp_bridge.RelayVSRBridge.upsampling_value(scale)
            with self.subTest(method=method):
                bridge = wgp_bridge.RelayVSRBridge({}, object())
                self.assertTrue(bridge.is_upsampling(method))
                self.assertEqual(bridge.scale_for_upsampling(method), scale)
                validate_methods(spatial=method, check_runtime=False)
                project = create_editor_project(name="Relay export")
                project["export"]["spatial_upsampling"] = method
                self.assertEqual(
                    normalize_editor_project(project)["export"]["spatial_upsampling"],
                    method,
                )
                record = processing_record(spatial=method)
                self.assertEqual(record["method"], method)
                self.assertEqual(record["multiplier"], scale)
                self.assertEqual(record["method_label"], "RelayVSR")

    def test_default_and_optional_native_model_selection(self):
        default = wgp_bridge.RelayVSRBridge({}, object())
        self.assertEqual([default.native_scale(f"relayvsr{scale:g}") for scale in (1, 1.5, 2)], [2, 2, 2])
        self.assertEqual(default.native_scale("relayvsr2.5"), 4)

        x4 = wgp_bridge.RelayVSRBridge({"services": {"relayvsr_x2_model": "x4"}}, object())
        self.assertEqual(x4.native_scale("relayvsr2"), 4)
        self.assertEqual(x4.native_scale("relayvsr3"), 4)

    def test_download_defs_pin_official_revision_and_expected_files(self):
        expected = {
            2: ("RelayVSR_X2_keyframes_int8_convrot.safetensors", "RelayVSR_X2_decoder_bf16.safetensors"),
            4: ("RelayVSR_keyframes_int8_convrot.safetensors", "RelayVSR_decoder_bf16.safetensors"),
        }
        for native, filenames in expected.items():
            with self.subTest(native=native):
                definition = wgp_bridge.RelayVSRBridge.download_def(native)
                self.assertEqual(definition["repoId"], "DeepBeepMeep/Wan2.1")
                self.assertEqual(definition["revision"], wgp_bridge.WEIGHTS_REVISION)
                self.assertEqual(definition["sourceFolderList"], ["RelayVSR"])
                self.assertEqual(tuple(definition["fileList"][0]), filenames)

    def test_paths_require_pinned_size_and_sha256(self):
        class Locator:
            def __init__(self, root):
                self.root = Path(root)

            def locate_file(self, relative, **_kwargs):
                return str(self.root / relative)

        original_metadata = wgp_bridge.ASSET_METADATA
        original_verified = set(wgp_bridge._VERIFIED_ASSETS)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            file_k, file_d = wgp_bridge.RelayVSRBridge.filenames(2)
            paths = [root / "RelayVSR" / name for name in (file_k, file_d)]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
            payloads = [b"keyframe weights", b"decoder weights"]
            metadata = {}
            for path, payload in zip(paths, payloads):
                path.write_bytes(payload)
                metadata[path.name] = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            wgp_bridge.ASSET_METADATA = metadata
            wgp_bridge._VERIFIED_ASSETS.clear()
            bridge = wgp_bridge.RelayVSRBridge({}, Locator(root))
            self.assertEqual(tuple(map(Path, bridge._paths(2))), tuple(paths))

            paths[0].write_bytes(b"corrupt weights")
            wgp_bridge._VERIFIED_ASSETS.clear()
            with self.assertRaisesRegex(RuntimeError, "size mismatch|SHA-256 mismatch"):
                bridge._paths(2)
        wgp_bridge.ASSET_METADATA = original_metadata
        wgp_bridge._VERIFIED_ASSETS.clear()
        wgp_bridge._VERIFIED_ASSETS.update(original_verified)

    def test_unknown_scale_is_rejected(self):
        bridge = wgp_bridge.RelayVSRBridge({}, object())
        with self.assertRaises(ValueError):
            bridge.native_scale("relayvsr2.25")
        with self.assertRaises(ValueError):
            validate_methods(spatial="relayvsr2.25", check_runtime=False)


if __name__ == "__main__":
    unittest.main()
