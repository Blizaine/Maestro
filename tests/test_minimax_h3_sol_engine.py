"""Regression coverage for Maestro's optional MiniMax H3 Sol Engine path."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))


class TestSolEngineSourceContracts(unittest.TestCase):
    def test_h3_declares_sol_without_exposing_it_as_a_global_backend(self):
        handler = (APP / "models" / "minimax_h3" / "minimax_h3_handler.py").read_text(encoding="utf-8")
        launch = (APP / "launch.py").read_text(encoding="utf-8")
        engine = (APP / "wgp.py").read_text(encoding="utf-8")
        attention = (APP / "shared" / "attention.py").read_text(encoding="utf-8")

        self.assertIn('"sol_attention": True', handler)
        self.assertIn('"sol_attention_status": _sol_attention_status', launch)
        self.assertIn('attn == "sol" and not model_def.get("sol_attention"', engine)
        self.assertIn("get_supported_override_attention_modes", engine)
        self.assertIn("def get_override_attention_modes", attention)
        self.assertNotIn('ret.append("sol")', attention)

    def test_sol_package_and_upstream_license_are_bundled(self):
        package = APP / "shared" / "sol_attn"
        self.assertTrue((package / "interface.py").is_file())
        self.assertTrue((package / "saganaki" / "LICENSE").is_file())
        self.assertIn(
            "Apache License",
            (package / "saganaki" / "LICENSE").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "SPDX-License-Identifier: Apache-2.0",
            (package / "interface.py").read_text(encoding="utf-8"),
        )

    def test_legacy_runtime_is_preserved_and_sol_runtime_is_opt_in(self):
        profile = (ROOT / "launcher_profile.js").read_text(encoding="utf-8")
        torch_script = (ROOT / "torch.js").read_text(encoding="utf-8")
        sol_script = (ROOT / "sol_torch.js").read_text(encoding="utf-8")
        menu = (ROOT / "pinokio.js").read_text(encoding="utf-8")
        reset = (ROOT / "reset.js").read_text(encoding="utf-8")

        self.assertIn('env: "env-sol"', profile)
        self.assertIn('target === "sm_89"', profile)
        self.assertIn("triton-windows==3.3.1.post19", torch_script)
        self.assertIn("triton-windows==3.6.0.post25", torch_script)
        self.assertIn("triton-windows==3.6.0.post25", sol_script)
        self.assertIn("torch==2.10.0", sol_script)
        self.assertIn("Start with H3 Sol Engine", menu)
        self.assertIn('path: "app/env-sol"', reset)

    def test_update_repairs_an_interrupted_optional_sol_runtime(self):
        updater = (ROOT / "update.js").read_text(encoding="utf-8")
        sol_script = (ROOT / "sol_torch.js").read_text(encoding="utf-8")

        self.assertIn("optionalSolReady", updater)
        self.assertIn("exists('${solRuntime.marker}')", updater)
        self.assertIn("exists('${solRuntime.flashMarker}')", updater)
        self.assertIn('uri: "sol_torch.js"', updater)
        self.assertIn("flash_only: true", updater)
        self.assertIn('{{args && args.flash_only}}', sol_script)

    def test_runtime_preflight_reports_sol_readiness(self):
        preflight = (APP / "scripts" / "runtime_preflight.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("H3 Sol Engine=", preflight)
        self.assertIn("triton-windows", preflight)

    def test_sol_upstream_notice_is_bundled(self):
        notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

        self.assertIn("46031940ba8af5d18054217e571149579424c0b1", notice)
        self.assertIn("ComfyUI-sol-attn", notice)

    def test_sol_start_uses_required_url_capture_contract(self):
        start = (ROOT / "start_sol.js").read_text(encoding="utf-8")

        self.assertIn('path: "app"', start)
        self.assertIn('"event": "/(http:\\/\\/[0-9.:]+)/"', start)
        self.assertIn('url: "{{input.event[1]}}"', start)

    def test_ui_persists_and_strips_model_scoped_override(self):
        types = (ROOT / "ui" / "src" / "types" / "index.ts").read_text(encoding="utf-8")
        optimizations = (ROOT / "ui" / "src" / "components" / "Sidebar" / "MiniMaxH3Optimizations.tsx").read_text(encoding="utf-8")
        advanced = (ROOT / "ui" / "src" / "components" / "Sidebar" / "AdvancedSettings.tsx").read_text(encoding="utf-8")
        store = (ROOT / "ui" / "src" / "stores" / "useStore.ts").read_text(encoding="utf-8")

        self.assertIn("override_attention?: '' | 'sol'", types)
        self.assertIn("H3 Optimizations", optimizations)
        self.assertIn("Sol Engine", optimizations)
        self.assertIn("params.override_attention === 'sol'", optimizations)
        self.assertNotIn("modelOptions?.sol_attention && (", advanced)
        self.assertIn("delete params.override_attention", store)
        self.assertIn("p.override_attention === 'sol'", store)


class TestSolAttentionRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

    def test_main_h3_blocks_share_sol_policy_but_refiner_does_not(self):
        from models.minimax_h3.transformer import MiniMaxH3Transformer

        model = MiniMaxH3Transformer(
            hidden_size=8,
            num_layers=2,
            token_refiner_layers=1,
            num_attention_heads=1,
            attention_head_dim=8,
            ffn_dim=12,
            video_channels=2,
            audio_channels=3,
            patch_size=(1, 1, 1),
            text_dim=6,
            curve_grid=4,
            curve_dim=2,
            rope_freq_dim=1,
            dtype=self.torch.float32,
        )

        self.assertIs(model.blocks[0].attn.sol_attention, model.sol_attention)
        self.assertIs(model.blocks[1].attn.sol_attention, model.sol_attention)
        self.assertIsNone(model.token_refiner.blocks[0].attn.sol_attention)

    def test_attention_routes_eligible_call_through_policy(self):
        from models.minimax_h3.transformer import MiniMaxH3Attention

        torch = self.torch

        class Probe:
            def __init__(self):
                self.called = False

            def use_for_layer(self, tokens, attention_mask=None):
                return tokens == 4 and attention_mask is None

            def __call__(self, qkv_list, use_sol):
                self.called = use_sol
                query, key, value = qkv_list
                qkv_list.clear()
                return torch.nn.functional.scaled_dot_product_attention(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                ).transpose(1, 2)

        probe = Probe()
        attention = MiniMaxH3Attention(
            8,
            1,
            8,
            1e-5,
            torch.float32,
            sol_attention=probe,
        ).eval()
        with torch.inference_mode():
            output = attention(torch.randn(1, 4, 8))

        self.assertTrue(probe.called)
        self.assertEqual(tuple(output.shape), (1, 4, 8))

    def test_kernel_failure_stays_on_dense_fallback_for_process(self):
        from models.minimax_h3.sol_attention import MiniMaxH3SolAttention

        policy = MiniMaxH3SolAttention()
        policy.enabled = True
        policy._fallback("test failure")

        self.assertTrue(policy._runtime_failed)
        self.assertFalse(policy.enabled)


if __name__ == "__main__":
    unittest.main()
