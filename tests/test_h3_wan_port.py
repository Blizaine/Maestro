"""Behavioral regressions for the adapted Wan2GP H3 features."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import torch
from models.minimax_h3.packing import build_packed_sequence, build_row_timesteps
from models.minimax_h3.masked_edit import grouped_rows, grouped_timesteps, outpaint_location, outpaint_mask, snap_mask_to_patches
from models.minimax_h3.voice_audio import audio_request
from models.minimax_h3.audio_refinement import refine_audio, refinement_sigmas, without_loras, refinement_unavailable
from models.minimax_h3.scheduler import MiniMaxH3Scheduler


class H3VoiceTests(unittest.TestCase):
    def test_two_speakers_keep_native_dialogue_and_ordered_references(self):
        prompt, seconds, refs = audio_request("Speaker 1: Hello there.\nSpeaker 2: Good morning.", 14.4, ["a.wav", "b.wav"])
        self.assertEqual([r["path"] for r in refs], ["a.wav", "b.wav"])
        self.assertIn("Speaker (S2) says once", prompt)
        self.assertIn("<d>[English] Good morning.</d>", prompt)
        self.assertEqual(seconds, 14.4)
        self.assertEqual(audio_request(prompt, 14.4, ["a.wav", "b.wav"])[0], prompt)

    def test_soundscape_and_duration_boundaries(self):
        self.assertNotIn("<d>", audio_request("Sound: Rain on a tin roof.", 5)[0])
        self.assertEqual(audio_request("Hello.", 45)[1], 45)
        for value in (4.9, 45.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audio_request("Hello.", value)

    def test_audio_preset_keeps_hidden_canvas_and_full_duration(self):
        from models.minimax_h3.minimax_h3_handler import family_handler
        definition = family_handler.query_model_def("minimax_h3_voice_audio", {})
        settings = {"duration_seconds": 15, "resolution": "864x480", "video_length": 345}
        family_handler.fix_settings("minimax_h3_voice_audio", 2.58, definition, settings)
        self.assertEqual(settings["resolution"], "32x32")
        self.assertGreaterEqual(settings["video_length"], 360)
        self.assertTrue(definition["audio_only"])
        self.assertFalse(definition["sliding_window"])


class H3MaskTests(unittest.TestCase):
    def test_short_multiwindow_outpaint_keeps_exact_total_frame_count(self):
        import ast
        from models.minimax_h3.minimax_h3_handler import family_handler
        definition = family_handler.query_model_def("minimax_h3", {})
        settings = {"video_length": 230, "sliding_window_size": 124,
            "sliding_window_overlap": 18, "minimax_h3_multi_window": True,
            "video_guide_outpainting": "0 0 25 25", "resolution": "896x448"}
        family_handler.fix_settings("minimax_h3", 2.58, definition, settings)
        self.assertEqual(settings["resolution"], "896x448")
        with patch("models.minimax_h3.minimax_h3_handler.apply_h3_window_memory_policy", return_value=None):
            self.assertIsNone(family_handler.validate_generative_settings("minimax_h3", definition, settings))
        self.assertEqual(settings["video_length"], 230)
        source = Path(__file__).resolve().parents[1] / "app" / "wgp.py"
        function = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name == "normalize_model_total_frame_count")
        align = Mock(return_value=243)
        namespace = {"align_model_frame_count": align}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        normalize = namespace["normalize_model_total_frame_count"]
        self.assertEqual(normalize(230, definition, window_size=124), 230)
        align.assert_not_called()
        self.assertEqual(normalize(230, definition, window_size=345), 243)

    def test_canvas_is_aligned_and_only_border_is_editable(self):
        margins = [10, 20, 13, 27]
        h, w, top, left = outpaint_location(480, 864, margins)
        self.assertTrue(all(value % 32 == 0 for value in (h, w, top, left)))
        mask = outpaint_mask(torch.zeros(3, 5, 480, 864), margins)
        self.assertEqual(mask[:, :, top:top+h, left:left+w].count_nonzero().item(), 0)
        self.assertGreater(mask.count_nonzero().item(), 0)
        self.assertLessEqual(top+h, 480)
        self.assertLessEqual(left+w, 864)

    def test_grouping_preserves_order_and_conditions_fixed_rows(self):
        rows = torch.tensor([[1., 0.], [0., 0.], [1., 1.], [0., 0.]])
        order, inverse, count = grouped_rows(rows)
        self.assertEqual(count, 2)
        torch.testing.assert_close(rows[order][inverse], rows)
        layout = build_packed_sequence(torch.ones(2, dtype=torch.long), 1, 4, 4, 2, (1, 2, 2))
        times, indices = build_row_timesteps(layout, .2, .3, .999, 1.)
        times, indices = grouped_timesteps(times, indices, layout, count)
        actual = times[indices][layout.video_indices]
        torch.testing.assert_close(actual, torch.tensor([.999, .999, .2, .2]))

    def test_mask_cannot_assign_two_noise_levels_to_one_patch(self):
        mask = torch.zeros(1, 1, 2, 4, 4)
        mask[0, 0, 1, 0, 0] = 1
        snapped = snap_mask_to_patches(mask, (1, 2, 2))
        self.assertEqual(snapped.sum().item(), 4)
        self.assertEqual(snapped[0, 0, 1, :2, :2].sum().item(), 4)

    def test_grouped_transformer_restores_spatial_order(self):
        from models.minimax_h3.transformer import MiniMaxH3Transformer
        torch.manual_seed(1)
        model = MiniMaxH3Transformer(hidden_size=32, num_layers=1, num_attention_heads=2,
            attention_head_dim=16, ffn_dim=64, text_dim=32, curve_grid=None,
            curve_dim=8, timestep_input_dim=16, time_embed_hidden_size=32, rope_freq_dim=2,
            dtype=torch.float32).eval()
        layout = build_packed_sequence(torch.ones(2, dtype=torch.long), 2, 4, 4, 2, (1, 2, 2))
        times, indices = build_row_timesteps(layout, .2, .3, .999, 1.)
        kwargs = dict(hidden_states=torch.randn(1, 8, 96), audio_hidden_states=torch.randn(1, 4, 32),
            encoder_hidden_states=torch.randn(1, 2, 32), timestep=times, timestep_indices=indices,
            token_tags=layout.token_tags, position_ids=layout.position_ids, video_indices=layout.video_indices,
            audio_indices=layout.audio_indices, text_indices=layout.text_indices, return_dict=False)
        order = torch.tensor([1, 3, 5, 7, 0, 2, 4, 6])
        with torch.inference_mode():
            baseline = model(**kwargs)
            grouped = model(**kwargs, target_video_order=order, target_video_inverse_order=torch.argsort(order))
        for expected, actual in zip(baseline, grouped):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


class H3RefinementTests(unittest.TestCase):
    def test_adapters_restore_on_exception(self):
        transformer = torch.nn.Linear(2, 2)
        transformer._loras_active_adapters = ["turbo"]
        transformer._loras_scaling = {"turbo": "0.8"}
        transformer._lora_step_no = 4
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with without_loras(transformer):
                self.assertEqual(transformer._loras_active_adapters, [])
                raise RuntimeError("synthetic")
        self.assertEqual(transformer._loras_active_adapters, ["turbo"])
        self.assertEqual(transformer._loras_scaling, {"turbo": "0.8"})
        self.assertEqual(transformer._lora_step_no, 4)

    def test_refinement_runs_six_audio_steps_without_mutating_video_or_sources(self):
        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
            def forward(self, **kwargs):
                self.calls.append(kwargs)
                self.assertions(kwargs)
                return (torch.ones_like(kwargs["hidden_states"])*999,
                        torch.ones_like(kwargs["audio_hidden_states"])*.1)
        transformer = Transformer()
        def check(kwargs):
            self.assertEqual(transformer._loras_active_adapters, [])
            self.assertEqual(kwargs["packed_layout"].num_condition_video_rows, 0)
            self.assertEqual(kwargs["packed_layout"].num_condition_audio_rows, 0)
        transformer.assertions = check
        conditioner = SimpleNamespace(forward_ref2va=Mock(return_value=(torch.zeros(1, 2, 8), torch.ones(2, dtype=torch.long))))
        model = SimpleNamespace(transformer=transformer, conditioner=conditioner, device="cpu", _interrupt=False,
            omni_reference=True, scheduler=MiniMaxH3Scheduler(shift=12), audio_scheduler=MiniMaxH3Scheduler(shift=3), patch_size=(1, 2, 2))
        layout = build_packed_sequence(torch.ones(2, dtype=torch.long), 2, 4, 4, 2, (1, 2, 2),
            ("first",), audio_condition_anchors=(("first", 1),))
        video = torch.randn(layout.video_indices.numel(), 96)
        audio = torch.randn(layout.audio_indices.numel(), 32)
        original_video, original_audio = video.clone(), audio.clone()
        output = refine_audio(model, video, audio, layout, "prompt", (2, 4, 4), 2, 42)
        self.assertEqual(len(transformer.calls), 6)
        conditioner.forward_ref2va.assert_called_once_with("prompt", "cpu", [])
        self.assertTrue(torch.equal(video, original_video))
        self.assertTrue(torch.equal(audio, original_audio))
        self.assertTrue(torch.equal(output[:layout.num_condition_audio_rows], audio[:layout.num_condition_audio_rows]))
        self.assertFalse(torch.equal(output[layout.num_condition_audio_rows:], audio[layout.num_condition_audio_rows:]))
        self.assertEqual(refinement_sigmas(1).numel(), 7)
        self.assertAlmostEqual(refinement_sigmas(1)[0].item(), .5)

    def test_exclusions(self):
        for flag in ("audio_only", "pdd", "fused", "source_audio"):
            self.assertTrue(refinement_unavailable(**{flag: True}))
        self.assertEqual(refinement_unavailable(), "")


class H3VDNTests(unittest.TestCase):
    def test_shipped_profiles_are_available_as_studio_presets(self):
        import ast, json, os
        path = Path(__file__).resolve().parents[1] / "app" / "launch.py"
        function = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name == "_bundled_generation_presets")
        namespace = {"__file__": str(path), "os": os, "json": json}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        presets = namespace["_bundled_generation_presets"]()
        for model in ("minimax_h3_vdn", "minimax_h3_vdn_full"):
            recipes = {p["name"]: p for p in presets if p["model_type"] == model}
            turbo = recipes["VDN Turbo 8 Steps"]
            self.assertTrue(turbo["builtin"])
            self.assertEqual(turbo["params"]["num_inference_steps"], 8)
            self.assertIn("/304d34f7751f8ba9ca0eb55d5d10044234cdbfe2/", turbo["activated_loras"][0])
            self.assertEqual(recipes["VDN Standard 50 Steps"]["activated_loras"], [])

    def test_adapter_url_can_download_without_a_prior_classic_ui_cache(self):
        import ast, os
        path = Path(__file__).resolve().parents[1] / "app" / "wgp.py"
        function = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name == "check_loras_exist")
        download = Mock()
        target = str(Path("missing-vdn-turbo-test.safetensors").resolve())
        namespace = {"os": os, "_ensure_loras_url_cache": lambda: None,
            "get_lora_dir": lambda model: ".", "resolve_lora_path": lambda model, name: target,
            "loras_url_cache": {}, "download_file": download}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        url = "https://huggingface.co/example/model/resolve/pinned/adapter.safetensors"
        self.assertEqual(namespace["check_loras_exist"]("minimax_h3_vdn", [url], download=True), "")
        download.assert_called_once_with(url, target)

    def test_diffusers_adapter_keeps_qkv_and_swiglu_math(self):
        from models.minimax_h3.lora_vdn import normalize_diffusers_lora
        model = torch.nn.Module()
        model.use_adaln_curves = False
        attention = torch.nn.Module()
        attention.qkv_proj = torch.nn.Linear(4, 12, bias=False)
        block = torch.nn.Module()
        block.attn = attention
        model.blocks = torch.nn.ModuleList([block])
        source, products = {}, []
        torch.manual_seed(7)
        for name in ("q", "k", "v"):
            down, up = torch.randn(2, 4), torch.randn(4, 2)
            source[f"transformer_blocks.0.attn.to_{name}.lora_A.turbo.weight"] = down
            source[f"transformer_blocks.0.attn.to_{name}.lora_B.turbo.weight"] = up
            products.append(up @ down)
        up = torch.arange(16).reshape(8, 2)
        source["transformer_blocks.0.ff.net.0.proj.lora_B.turbo.weight"] = up
        result = normalize_diffusers_lora(source, model)
        torch.testing.assert_close(result["blocks.0.attn.qkv_proj.lora_B.weight"] @
            result["blocks.0.attn.qkv_proj.lora_A.weight"], torch.cat(products))
        torch.testing.assert_close(result["blocks.0.mlp.fc1.lora_B.weight"], torch.cat((up[4:], up[:4])))

    def test_vdn_module_does_not_reorder_absent_base_qkv(self):
        from models.minimax_h3.minimax_h3_main import _strip_transformer_wrappers
        source = {"transformer_blocks.0.attn.linear_attention.alpha.down.comfy_quant":
                  torch.tensor(list(b'{"convrot":true}'), dtype=torch.uint8)}
        converted, _, _ = _strip_transformer_wrappers(source)
        self.assertIn("blocks.0.attn.vdn.linear_attention.alpha.down.comfy_quant", converted)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_triton_temporal_convolution_matches_torch(self):
        from models.minimax_h3.vdn_attention import LinearAttentionSepConv, triton
        if triton is None:
            self.skipTest("Triton unavailable")
        torch.manual_seed(3)
        model = LinearAttentionSepConv(32, dtype=torch.float32, device="cuda")
        tokens = torch.randn(11*4, 2, 16, device="cuda")
        with torch.inference_mode():
            for name in ("k", "v"):
                actual = model.apply(name, tokens, 11, (2, 2), True)
                expected = model.apply(name, tokens, 11, (2, 2), False)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
