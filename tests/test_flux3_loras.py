"""CPU-only contracts for FLUX 3's public adapter naming compatibility."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.flux3.dit import DiT  # noqa: E402
from models.flux3.flux3_handler import MODEL_TYPE  # noqa: E402


class Flux3LoraTests(unittest.TestCase):
    def test_public_diffusers_safetensors_map_to_native_image_text_and_single_blocks(self):
        source = {
            "base_model.model.transformer.mode_blocks.image.0.attn.to_q.lora_A.default.weight": torch.arange(8).reshape(2, 4),
            "transformer.mode_blocks.image.0.attn.to_out.0.lora_B.weight": torch.arange(8).reshape(4, 2),
            "diffusion_model.mode_blocks.image_cond.3.ff.linear_in.lora_down.weight": torch.ones((2, 4)),
            "single_transformer_blocks.0.attn.to_v.lora_up.weight": torch.ones((4, 2)),
            "txt_mode_blocks.1.ff.linear_out.lora_A.weight": torch.ones((2, 4)),
            "context_embedder.lora_A.weight": torch.ones((2, 4)),
            "x_embedders.image.lora_A.weight": torch.ones((2, 4)),
            "proj_out.lora_B.weight": torch.ones((4, 2)),
            # Existing Kohya checkpoints flatten target module dots to underscores.
            "lora_unet_single_blocks_0_v_proj.lora_down.weight": torch.ones((2, 4)),
            # Native split projections can also load adapters saved for fused QKV.
            "content_mode_blocks.image.0.qkv.lora_A.weight": torch.ones((2, 4)),
            # Flux 2's transformer_blocks layout must not be rewritten as Klein.
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones((2, 4)),
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            filename = str(Path(directory) / "public_flux3_adapter.safetensors")
            save_file(source, filename)
            with safe_open(filename, framework="pt", device="cpu") as adapter:
                fixture = {key: adapter.get_tensor(key).clone() for key in adapter.keys()}
        with torch.device("meta"):
            distilled_dit = DiT(flow_map=True)
            base_dit = DiT(flow_map=False)

        converted_distilled = distilled_dit.preprocess_loras(MODEL_TYPE, fixture)
        converted_base = base_dit.preprocess_loras("flux3_klein_base", fixture)
        expected = {
            "content_mode_blocks.image.0.q_proj.lora_A.weight": "base_model.model.transformer.mode_blocks.image.0.attn.to_q.lora_A.default.weight",
            "content_mode_blocks.image.0.attn_out.lora_B.weight": "transformer.mode_blocks.image.0.attn.to_out.0.lora_B.weight",
            "content_mode_blocks.image_cond.3.mlp_in.lora_down.weight": "diffusion_model.mode_blocks.image_cond.3.ff.linear_in.lora_down.weight",
            "single_blocks.0.v_proj.lora_up.weight": "single_transformer_blocks.0.attn.to_v.lora_up.weight",
            "txt_mode_blocks.1.mlp_out.lora_A.weight": "txt_mode_blocks.1.ff.linear_out.lora_A.weight",
            "txt_in.lora_A.weight": "context_embedder.lora_A.weight",
            "emb_in.image.lora_A.weight": "x_embedders.image.lora_A.weight",
            "final_layer.image.linear.lora_B.weight": "proj_out.lora_B.weight",
            "single_blocks.0.v_proj.lora_down.weight": "lora_unet_single_blocks_0_v_proj.lora_down.weight",
            "content_mode_blocks.image.0.qkv.lora_A.weight": "content_mode_blocks.image.0.qkv.lora_A.weight",
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
        }
        self.assertEqual(set(converted_distilled), set(expected))
        self.assertEqual(converted_distilled, converted_base)
        for target, fixture_key in expected.items():
            torch.testing.assert_close(converted_distilled[target], fixture[fixture_key])

    def test_dit_declares_the_fused_qkv_split_layout_for_generic_lora_loading(self):
        with torch.device("meta"):
            dit = DiT()
        self.assertEqual(
            dit.split_linear_modules_map["qkv"],
            {"mapped_modules": ["q_proj", "k_proj", "v_proj"], "split_sizes": [3072, 3072, 3072]},
        )

    def test_colliding_wrapped_keys_are_rejected(self):
        with torch.device("meta"):
            dit = DiT()
        down = torch.ones((2, 4))
        with self.assertRaisesRegex(ValueError, "conversion collision"):
            dit.preprocess_loras(MODEL_TYPE, {
                "transformer.mode_blocks.image.0.attn.to_q.lora_A.weight": down,
                "content_mode_blocks.image.0.q_proj.lora_A.weight": down,
            })


if __name__ == "__main__":
    unittest.main()
