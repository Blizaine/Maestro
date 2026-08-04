"""Model-free and optional runtime regressions for MiniMax H3 support."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_APP = _ROOT / "app"
_HANDLER_PATH = _APP / "models" / "minimax_h3" / "minimax_h3_handler.py"
_MAIN_PATH = _APP / "models" / "minimax_h3" / "minimax_h3_main.py"
_PACKING_PATH = _APP / "models" / "minimax_h3" / "packing.py"
_TRANSFORMER_PATH = _APP / "models" / "minimax_h3" / "transformer.py"
_CONDITIONER_PATH = _APP / "models" / "minimax_h3" / "conditioner.py"
_CHECKPOINT_PATH = _APP / "models" / "minimax_h3" / "checkpoint.py"
_NVFP4_PATH = _APP / "shared" / "qtypes" / "nvfp4.py"
_WGP_PATH = _APP / "wgp.py"
_LAUNCH_PATH = _APP / "launch.py"
_LLM_SERVICE_PATH = _APP / "services" / "llm_service.py"
_DEFAULT_PATH = _APP / "defaults" / "minimax_h3.json"
_STORE_PATH = _ROOT / "ui" / "src" / "stores" / "useStore.ts"
_PROMPT_INPUT_PATH = _ROOT / "ui" / "src" / "components" / "Sidebar" / "PromptInput.tsx"
_ENHANCE_GUIDES_PATH = _APP / "services" / "enhance_guides.py"
_PROMPT_POLISH_PATH = _APP / "services" / "director" / "prompt_polish.py"
_H3_ENHANCE_GUIDE_PATH = _APP / "services" / "llm_guides" / "enhance" / "minimax_h3_video.md"
_H3_DIALECT_GUIDE_PATH = _APP / "services" / "llm_guides" / "dialect" / "minimax_h3_video.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_handler_class():
    tree = ast.parse(_read(_HANDLER_PATH), filename=str(_HANDLER_PATH))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if any(name.startswith("_") for name in names):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "_hf_url":
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "family_handler":
            selected.append(node)
    namespace = {
        "os": os,
        "torch": types.SimpleNamespace(bfloat16="bfloat16"),
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_HANDLER_PATH), "exec"), namespace)
    return namespace["family_handler"]


def _load_frame_aligner():
    tree = ast.parse(_read(_WGP_PATH), filename=str(_WGP_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "align_model_frame_count"
    )
    namespace = {}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_WGP_PATH), "exec"), namespace)
    return namespace["align_model_frame_count"]


class TestMiniMaxH3Definition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler = _load_handler_class()

    def test_default_model_is_pinned_and_consumer_friendly(self):
        defaults = json.loads(_DEFAULT_PATH.read_text(encoding="utf-8"))
        model = defaults["model"]
        self.assertEqual(model["architecture"], "minimax_h3")
        self.assertEqual(defaults["num_inference_steps"], 20)
        self.assertEqual(defaults["video_length"], 124)
        self.assertEqual(defaults["resolution"], "864x480")
        self.assertIn("minimax_h3_fl2va_pruned_fp8_scaled.safetensors", model["URLs"][0])
        self.assertIn("0543966fbdce5ba05709a8f2031c94bdba629b4a", model["URLs"][0])

    def test_handler_exposes_base_fl2va_contract(self):
        model_def = self.handler.query_model_def("minimax_h3", {})
        self.assertEqual(self.handler.query_supported_types(), ["minimax_h3"])
        self.assertEqual((model_def["fps"], model_def["frames_minimum"]), (24, 124))
        self.assertEqual((model_def["frames_steps"], model_def["frames_maximum"]), (17, 345))
        self.assertEqual(
            (model_def["frame_alignment_modulus"], model_def["frame_alignment_remainder"]),
            (17, 5),
        )
        self.assertEqual(model_def["image_prompt_types_allowed"], "TSE")
        self.assertTrue(model_def["end_frames_always_enabled"])
        self.assertTrue(model_def["t2v_class"])
        self.assertTrue(model_def["i2v_class"])
        self.assertTrue(model_def["returns_audio"])
        self.assertTrue(model_def["no_negative_prompt"])
        self.assertFalse(model_def["sliding_window"])
        self.assertIn("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", model_def["text_encoder_URLs"][0])

    def test_h3_reserves_transformer_activation_workspace(self):
        source = _read(_HANDLER_PATH)
        self.assertIn("_TRANSFORMER_WORKING_VRAM_MB = 10 * 1024", source)
        self.assertIn('"workingVRAM": {', source)
        self.assertIn('"transformer": _TRANSFORMER_WORKING_VRAM_MB', source)

    def test_all_auxiliary_downloads_are_revision_pinned(self):
        downloads = self.handler.query_model_files(lambda item: [item], "minimax_h3")
        self.assertEqual(len(downloads), 2)
        self.assertEqual(downloads[0]["repoId"], "Comfy-Org/MiniMax-H3")
        self.assertEqual(downloads[0]["revision"], "0543966fbdce5ba05709a8f2031c94bdba629b4a")
        self.assertEqual(downloads[0]["sourceFolderList"], ["vae"])
        self.assertIn("minimax_h3_video_vae_fp16.safetensors", downloads[0]["fileList"][0])
        self.assertIn("minimax_h3_audio_vae_fp32.safetensors", downloads[0]["fileList"][0])
        self.assertEqual(downloads[1]["repoId"], "MiniMaxAI/MiniMax-H3")
        self.assertEqual(downloads[1]["revision"], "5d9b308a59ab12e67147f191e184baf704185bd1")

    def test_maestro_registers_the_family_and_uses_its_native_frame_grid(self):
        source = _read(_WGP_PATH)
        self.assertIn('"models.minimax_h3.minimax_h3_handler"', source)
        self.assertIn("video_length = align_model_frame_count(video_length, model_def)", source)
        self.assertIn(
            "frame_num=align_model_frame_count(current_video_length, model_def, for_generation=True)",
            source,
        )
        self.assertIn('model_def.get("frames_maximum", None)', source)

    def test_h3_is_enabled_for_existing_and_fresh_installs(self):
        store = _read(_STORE_PATH)
        default_block = store.split("const DEFAULT_ENABLED_MODELS = new Set([", 1)[1].split("])\n", 1)[0]
        self.assertIn("'minimax_h3'", default_block)
        self.assertIn("const DEFAULTS_VERSION = 6", store)
        self.assertIn("6: ['minimax_h3']", store)
        self.assertIn('md.get("returns_audio", False)', _read(_LAUNCH_PATH))

    def test_h3_prompt_guides_cover_native_audio_and_director(self):
        self.assertIn('"minimax_h3": "minimax_h3_video.md"', _read(_ENHANCE_GUIDES_PATH))
        self.assertIn('"minimax_h3": "minimax_h3_video"', _read(_PROMPT_POLISH_PATH))
        enhance_guide = _read(_H3_ENHANCE_GUIDE_PATH)
        dialect_guide = _read(_H3_DIALECT_GUIDE_PATH)
        self.assertIn("joint video-and-audio", enhance_guide)
        for required in (
            "integrated_multimodal_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
            "(S1)",
            "<d>[English]",
            "remain silent with their mouths closed",
        ):
            self.assertIn(required, enhance_guide)
        self.assertIn("but supplies no script", enhance_guide)
        self.assertIn("<d>[English] Exact words.</d>", dialect_guide)
        self.assertIn("Never invent extra speech", dialect_guide)

    def test_h3_enhance_path_preserves_context_ir_contract(self):
        launch = _read(_LAUNCH_PATH)
        llm_service = _read(_LLM_SERVICE_PATH)
        self.assertIn("needs_h3_context_ir", launch)
        self.assertIn("enhancer_enabled > 0 and not needs_h3_context_ir", launch)
        self.assertIn("is_h3_context_ir", llm_service)
        self.assertIn('mode in ("video", "avatar") and not is_h3_context_ir', llm_service)
        self.assertIn("CRITICAL MINIMAX H3 OUTPUT CONTRACT", llm_service)
        self.assertIn("effective_max_tokens = max(effective_max_tokens, 768)", llm_service)

    def test_non_sliding_h3_enhance_request_stays_one_timeline(self):
        store = _read(_STORE_PATH)
        prompt_input = _read(_PROMPT_INPUT_PATH)
        expected = "supportsSlidingWindows = state.modelOptions?.sliding_window === true"
        self.assertIn(expected, store)
        self.assertIn("supportsSlidingWindows && stride > 0", store)
        self.assertIn("supportsSlidingWindows = modelOptions?.sliding_window === true", prompt_input)
        self.assertIn("supportsSlidingWindows && stride > 0", prompt_input)

    def test_frame_aligner_preserves_h3_and_legacy_grids(self):
        align = _load_frame_aligner()
        h3 = {
            "frames_minimum": 124,
            "frames_maximum": 345,
            "frame_alignment_modulus": 17,
            "frame_alignment_remainder": 5,
            "frame_alignment_mode": "ceil",
            "latent_size": 17,
        }
        self.assertEqual([align(value, h3) for value in (1, 120, 124, 125, 345, 999)], [124, 124, 124, 141, 345, 345])
        legacy = {"latent_size": 4, "frames_steps": 4}
        self.assertEqual(align(120, legacy), 117)
        self.assertEqual(align(120, legacy, for_generation=True), 121)


class TestMiniMaxH3RuntimeSource(unittest.TestCase):
    def test_runtime_uses_the_official_dual_scheduler_and_audio_output(self):
        main = _read(_MAIN_PATH)
        self.assertIn("MiniMaxH3Scheduler(shift=12.0)", main)
        self.assertIn("MiniMaxH3Scheduler(shift=3.0)", main)
        self.assertIn("audio_sampling_rate\": 32000", main)
        self.assertIn("MINIMAX_H3_KEYFRAME_ENCODE_SEED", main)
        self.assertIn("prepare_keyframe_image", main)

    def test_consumer_checkpoint_shapes_are_kept_native(self):
        transformer = _read(_TRANSFORMER_PATH)
        conditioner = _read(_CONDITIONER_PATH)
        self.assertIn("self.qkv_proj", transformer)
        self.assertIn("self.fc1", transformer)
        self.assertIn("adaln_t_table", transformer)
        self.assertIn("curve_dim: int = 8", transformer)
        self.assertIn("TEXT_ENCODER_LAYERS = 50", conditioner)
        self.assertIn("class MiniMaxH3Int8Embedding", conditioner)
        self.assertIn("pre_quant_scale", conditioner)
        self.assertIn("self.model.norm = nn.Identity()", conditioner)
        self.assertIn("attention_mask=attention_mask,", conditioner)
        self.assertIn("native causal attention", conditioner)
        self.assertIn("dtype=torch.float32", transformer)

    def test_compact_vae_adapters_and_nvfp4_awq_scale_are_present(self):
        checkpoint = _read(_CHECKPOINT_PATH)
        nvfp4 = _read(_NVFP4_PATH)
        self.assertIn("_reorder_interleaved_qkv", checkpoint)
        self.assertIn("weight_g", checkpoint)
        self.assertIn("weight_v", checkpoint)
        self.assertIn('qmodule.register_buffer(\n                "pre_quant_scale"', nvfp4)
        self.assertIn("input = input * pre_quant_scale.to", nvfp4)

    def test_conditioner_loader_preserves_mixed_quantization_contract(self):
        main = _read(_MAIN_PATH)
        checkpoint = _read(_CHECKPOINT_PATH)
        self.assertIn("preprocess_sd=preprocess_conditioner_state_dict", main)
        self.assertIn("with init_empty_weights(include_buffers=False):", main)
        self.assertIn('descriptor.get("format") != "int8_tensorwise"', checkpoint)
        self.assertIn('state_dict.pop(f"{prefix}.comfy_quant", None)', checkpoint)

    def test_h3_loaders_materialize_nonpersistent_runtime_buffers(self):
        main = _read(_MAIN_PATH)
        video_loader = main.split("def _load_video_vae", 1)[1].split("def _load_audio_vae", 1)[0]
        audio_loader = main.split("def _load_audio_vae", 1)[1].split("class MiniMaxH3Model", 1)[0]
        self.assertIn("init_empty_weights(include_buffers=False)", video_loader)
        self.assertIn("init_empty_weights(include_buffers=False)", audio_loader)

    def test_upstream_provenance_is_recorded(self):
        provenance = _read(_APP / "models" / "minimax_h3" / "UPSTREAM.md")
        self.assertIn("abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc", provenance)
        self.assertIn("5d9b308a59ab12e67147f191e184baf704185bd1", provenance)
        self.assertIn("0543966fbdce5ba05709a8f2031c94bdba629b4a", provenance)
        self.assertIn("Apache-2.0", provenance)


_RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "diffusers", "transformers")
)


@unittest.skipUnless(_RUNTIME_AVAILABLE, "MiniMax H3 runtime dependencies are not installed")
class TestMiniMaxH3RuntimeMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_APP))
        import torch

        cls.torch = torch

    @classmethod
    def tearDownClass(cls):
        if sys.path and sys.path[0] == str(_APP):
            sys.path.pop(0)

    def test_video_patch_round_trip_and_scheduler_length(self):
        from models.minimax_h3.packing import patchify_video_latents, unpatchify_video_tokens
        from models.minimax_h3.scheduler import MiniMaxH3Scheduler

        source = self.torch.arange(1 * 2 * 3 * 4 * 6, dtype=self.torch.float32).reshape(1, 2, 3, 4, 6)
        rows = patchify_video_latents(source, (1, 2, 2))
        restored = unpatchify_video_tokens(rows, 3, 4, 6, 2, (1, 2, 2))
        self.assertTrue(self.torch.equal(source, restored))

        scheduler = MiniMaxH3Scheduler(shift=12.0)
        scheduler.set_timesteps(20, device="cpu")
        self.assertEqual(len(scheduler.sigmas), 20)
        self.assertEqual(len(scheduler.timesteps), 19)
        self.assertEqual(float(scheduler.timesteps[0]), 0.0)
        self.assertEqual(float(scheduler.sigmas[-1]), 0.0)

    def test_keyframe_normalization_stays_on_cpu_with_non_cpu_default_device(self):
        from models.minimax_h3.minimax_h3_main import _keyframe_latent_stats_cpu

        previous_device = self.torch.get_default_device()
        try:
            # Maestro runs with a CUDA default device. ``meta`` reproduces the
            # constructor-routing behavior without requiring a GPU in CI.
            self.torch.set_default_device("meta")
            means, stds = _keyframe_latent_stats_cpu()
        finally:
            self.torch.set_default_device(previous_device)

        self.assertEqual(means.device.type, "cpu")
        self.assertEqual(stds.device.type, "cpu")
        self.assertEqual(tuple(means.shape), (1, 24, 1, 1, 1))
        self.assertEqual(tuple(stds.shape), (1, 24, 1, 1, 1))
        self.assertEqual(means.dtype, self.torch.float32)
        self.assertEqual(stds.dtype, self.torch.float32)

    def test_tiny_joint_transformer_forward(self):
        from models.minimax_h3.transformer import MiniMaxH3Transformer

        model = MiniMaxH3Transformer(
            hidden_size=8,
            num_layers=1,
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
        ).eval()
        # Production weights replace this table from the checkpoint.  The
        # tiny model has no checkpoint, so initialize its empty placeholder
        # to keep the numerical smoke test deterministic.
        model.adaln_t_table.data.zero_()
        self.assertEqual(model.video_patch_proj._lock_dtype, self.torch.float32)
        self.assertEqual(model.audio_patch_proj._lock_dtype, self.torch.float32)
        self.assertEqual(model.blocks[0].adaln_proj.linear._lock_dtype, self.torch.float16)
        self.assertEqual(model.final_layer.adaln_proj.linear._lock_dtype, self.torch.float16)
        self.assertEqual(model.final_layer.video_out._lock_dtype, self.torch.float32)
        self.assertEqual(model.final_layer.audio_out._lock_dtype, self.torch.float32)
        video_rows = self.torch.randn(1, 3, 2)
        audio_rows = self.torch.randn(1, 4, 3)
        text_rows = self.torch.randn(1, 2, 6)
        position_ids = self.torch.zeros(9, 3, dtype=self.torch.float64)
        token_tags = self.torch.tensor([1, 1, 2, 2, 2, 2, 0, 0, 0])
        timestep_indices = self.torch.tensor([0, 0, 1, 1, 1, 1, 0, 0, 0])
        video, audio = model(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=text_rows,
            timestep=self.torch.tensor([0.1, 0.4]),
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=self.torch.tensor([6, 7, 8]),
            audio_indices=self.torch.tensor([2, 3, 4, 5]),
            text_indices=self.torch.tensor([0, 1]),
            return_dict=False,
        )
        self.assertEqual(tuple(video.shape), (1, 3, 2))
        self.assertEqual(tuple(audio.shape), (1, 4, 3))
        self.assertTrue(self.torch.isfinite(video).all())
        self.assertTrue(self.torch.isfinite(audio).all())

    def test_chunked_h3_projections_match_unchunked_math(self):
        import models.minimax_h3.transformer as h3_transformer

        attention = h3_transformer.MiniMaxH3Attention(8, 1, 8, 1e-5, self.torch.float32).eval()
        mlp = h3_transformer.MiniMaxH3MLP(8, 12, self.torch.float32).eval()
        hidden = self.torch.randn(1, 7, 8)
        positions = self.torch.zeros(7, 3)
        rotary = h3_transformer.MiniMaxH3RotaryEmbedding(1)(positions)

        previous = h3_transformer.MINIMAX_H3_ACTIVATION_CHUNK_TOKENS
        try:
            with self.torch.inference_mode():
                h3_transformer.MINIMAX_H3_ACTIVATION_CHUNK_TOKENS = 64
                expected_attention = attention(hidden, rotary)
                expected_mlp = mlp(hidden)
                h3_transformer.MINIMAX_H3_ACTIVATION_CHUNK_TOKENS = 2
                actual_attention = attention(hidden, rotary)
                actual_mlp = mlp(hidden)
        finally:
            h3_transformer.MINIMAX_H3_ACTIVATION_CHUNK_TOKENS = previous

        self.assertTrue(self.torch.allclose(actual_attention, expected_attention, atol=1e-5, rtol=1e-5))
        self.assertTrue(self.torch.allclose(actual_mlp, expected_mlp, atol=1e-5, rtol=1e-5))

    def test_curve_adaln_uses_fp32_math_with_compact_fp16_storage(self):
        from models.minimax_h3.transformer import MiniMaxH3AdaLNProjection

        projection = MiniMaxH3AdaLNProjection(2, 2, 2, 1, self.torch.float16).eval()
        projection.linear.weight.data.copy_(
            self.torch.tensor(
                [[0.3333, -1.777], [2.125, 0.03125], [-0.8125, 1.333], [3.141, -2.718]],
                dtype=self.torch.float16,
            )
        )
        projection.linear.bias.data.copy_(
            self.torch.tensor([0.125, -0.25, 0.375, -0.5], dtype=self.torch.float16)
        )
        curve = self.torch.tensor([[0.12345, -0.98765]], dtype=self.torch.float32)
        chunks = projection(curve)
        actual = self.torch.cat(chunks, dim=-1)
        expected = self.torch.nn.functional.linear(
            curve,
            projection.linear.weight.float(),
            projection.linear.bias.float(),
        )

        self.assertEqual(projection.linear.weight.dtype, self.torch.float16)
        self.assertEqual(actual.dtype, self.torch.float32)
        self.assertTrue(self.torch.equal(actual, expected))

    def test_nvfp4_pre_quant_scale_loads_and_affects_forward(self):
        from models.minimax_h3.conditioner import MiniMaxH3PreScaledLinear
        from shared.qtypes.nvfp4 import QLinearNVFP4, _NVFP4_QTYPE

        source = MiniMaxH3PreScaledLinear(3, 2, bias=True, dtype=self.torch.float32)
        qmodule = QLinearNVFP4.qcreate(source, _NVFP4_QTYPE, device="cpu")
        qmodule.weight = self.torch.nn.Parameter(
            self.torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
        )
        qmodule.bias = self.torch.nn.Parameter(self.torch.tensor([0.25, -0.5]))

        scale = self.torch.tensor([2.0, 3.0, 4.0])
        missing_keys, unexpected_keys, error_messages = [], [], []
        state_dict = {"pre_quant_scale": scale.clone()}
        qmodule._load_from_state_dict(
            state_dict,
            "",
            {},
            False,
            missing_keys,
            unexpected_keys,
            error_messages,
        )
        self.assertTrue(self.torch.equal(qmodule.pre_quant_scale, scale))
        self.assertNotIn("pre_quant_scale", state_dict)

        input_rows = self.torch.tensor([[1.0, 1.0, 1.0]])
        expected = self.torch.nn.functional.linear(
            input_rows * scale,
            qmodule.weight,
            qmodule.bias,
        )
        self.assertTrue(self.torch.equal(qmodule(input_rows), expected))

        # MMGP's quant router transfers ordinary handler attributes but omits
        # registered buffers. Simulate that transfer and prove the mirrored
        # scale still governs the routed forward path.
        self.assertTrue(self.torch.equal(qmodule._nvfp4_pre_quant_scale, scale))
        del qmodule._buffers["pre_quant_scale"]
        self.assertFalse(hasattr(qmodule, "pre_quant_scale"))
        self.assertTrue(self.torch.equal(qmodule(input_rows), expected))

    def test_nvfp4_fallback_matches_official_combined_scale_order(self):
        from shared.qtypes.nvfp4 import (
            _NVFP4_LAYOUT_TENSORCORE,
            _dequantize_nvfp4_weight,
        )

        # TensorCore scale tiles require 128 output rows and 64 input
        # channels at minimum.  0xFF decodes to two -6.0 FP4 values.
        packed_weight = self.torch.full((128, 32), 0xFF, dtype=self.torch.uint8)
        block_scale = self.torch.full(
            (128, 4),
            0.00099945068359375,
            dtype=self.torch.bfloat16,
        )
        tensor_scale = self.torch.tensor(0.0030059814453125, dtype=self.torch.float32)
        actual = _dequantize_nvfp4_weight(
            packed_weight,
            block_scale,
            self.torch.ones((), dtype=self.torch.float32),
            tensor_scale,
            self.torch.bfloat16,
            self.torch.device("cpu"),
            layout=_NVFP4_LAYOUT_TENSORCORE,
        )
        expected_value = self.torch.tensor(-6.0, dtype=self.torch.bfloat16) * (
            block_scale[0, 0] * tensor_scale.to(self.torch.bfloat16)
        )
        old_order_value = (
            self.torch.tensor(-6.0, dtype=self.torch.bfloat16) * block_scale[0, 0]
        ) * tensor_scale.to(self.torch.bfloat16)

        self.assertTrue(self.torch.equal(actual, self.torch.full_like(actual, expected_value)))
        self.assertNotEqual(expected_value.item(), old_order_value.item())

    def test_row_scaled_int8_embedding_loads_and_dequantizes_selected_rows(self):
        from models.minimax_h3.checkpoint import preprocess_conditioner_state_dict
        from models.minimax_h3.conditioner import MiniMaxH3Int8Embedding

        weight = self.torch.tensor(
            [[1, -2, 3], [4, 5, -6], [-7, 8, 9], [10, -11, 12]],
            dtype=self.torch.int8,
        )
        scales = self.torch.tensor([0.5, 0.25, 2.0, 0.125], dtype=self.torch.float32)
        marker = self.torch.tensor(
            list(b'{"format":"int8_tensorwise"}'),
            dtype=self.torch.uint8,
        )
        state_dict = {
            "model.embed_tokens.comfy_quant": marker,
            "model.embed_tokens.weight": weight.clone(),
            "model.embed_tokens.weight_scale": scales.clone(),
        }
        processed = preprocess_conditioner_state_dict(state_dict)
        self.assertNotIn("model.embed_tokens.comfy_quant", processed)
        self.assertEqual(tuple(processed["model.embed_tokens.weight_scale"].shape), (4, 1))

        embedding = MiniMaxH3Int8Embedding(4, 3, None, self.torch.float32)
        embedding.load_state_dict(
            {
                "weight": processed["model.embed_tokens.weight"],
                "weight_scale": processed["model.embed_tokens.weight_scale"],
            },
            assign=True,
        )
        input_ids = self.torch.tensor([[3, 0, 2, 3]])
        expected = weight[input_ids].float() * scales[input_ids].unsqueeze(-1)
        self.assertTrue(self.torch.equal(embedding(input_ids), expected))
        self.assertFalse(embedding.weight.requires_grad)
        self.assertEqual(embedding._lock_dtype, self.torch.float32)


if __name__ == "__main__":
    unittest.main()
