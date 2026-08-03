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
_DEFAULT_PATH = _APP / "defaults" / "minimax_h3.json"
_STORE_PATH = _ROOT / "ui" / "src" / "stores" / "useStore.ts"
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
        self.assertIn("spoken dialogue", enhance_guide)
        self.assertIn("synchronized sound", dialect_guide)

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
        self.assertIn("pre_quant_scale", conditioner)
        self.assertIn("self.model.norm = nn.Identity()", conditioner)

    def test_compact_vae_adapters_and_nvfp4_awq_scale_are_present(self):
        checkpoint = _read(_CHECKPOINT_PATH)
        nvfp4 = _read(_NVFP4_PATH)
        self.assertIn("_reorder_interleaved_qkv", checkpoint)
        self.assertIn("weight_g", checkpoint)
        self.assertIn("weight_v", checkpoint)
        self.assertIn('qmodule.register_buffer(\n                "pre_quant_scale"', nvfp4)
        self.assertIn("input = input * pre_quant_scale.to", nvfp4)

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


if __name__ == "__main__":
    unittest.main()
