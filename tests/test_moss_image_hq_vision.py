"""CPU contracts for MOSS HQ v5's Qwen multimodal reference conditioning."""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from accelerate import init_empty_weights
from PIL import Image
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.moss_image.hq_vision import (  # noqa: E402
    Qwen3_5VisionPositionIndexer,
    encode_prompt_with_vision,
    load_hq_vision_components,
    restore_vision_rotary_buffers,
    vision_prompt_cache_key,
)
from models.moss_image import hq_vision as hq_vision_module  # noqa: E402
from models.moss_image.pipeline import MOSSImagePipeline  # noqa: E402
from models.moss_image.text_encoder import TextEncoder  # noqa: E402
from shared.prompt_enhancer.qwen3_5.configuration_qwen3_5 import (  # noqa: E402
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from shared.prompt_enhancer.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding  # noqa: E402
from shared.utils.text_encoder_cache import TextEncoderCache  # noqa: E402


class _MultimodalTokenizer:
    special_ids = {
        "<|image_pad|>": 5,
        "<|vision_start|>": 6,
        "<|vision_end|>": 7,
    }

    def __init__(self):
        self.rendered_messages = []

    def convert_tokens_to_ids(self, token):
        return self.special_ids.get(token, 1)

    def apply_chat_template(self, messages, **kwargs):
        self.rendered_messages.append((messages, kwargs))
        return messages[0]["content"]

    def __call__(self, rendered, **kwargs):
        ids = []
        tokens = sorted(self.special_ids, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(token) for token in tokens) + ")"
        for part in re.split(pattern, rendered):
            if not part:
                continue
            if part in self.special_ids:
                ids.append(self.special_ids[part])
            else:
                ids.extend([1] * max(1, len(part.split())))
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _RecordingTextEncoder(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, hidden_size)
        self.layers = nn.ModuleList()
        self.calls = []

    def forward(self, input_ids, inputs_embeds=None, position_ids=None, image_features=None, image_token_id=None):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if image_features is not None:
            image_mask = input_ids.eq(int(image_token_id))
            hidden = hidden.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(hidden),
                image_features.to(device=hidden.device, dtype=hidden.dtype).reshape(-1),
            )
        self.calls.append((input_ids.detach().clone(), hidden.detach().clone(), position_ids.detach().clone()))
        return hidden


class _FakeImageProcessor:
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
    patch_size = 16

    def __init__(self):
        self.sizes = []

    def __call__(self, *, images, return_tensors, size):
        self.sizes.append(size)
        values = [image.getpixel((0, 0))[0] for image in images]
        return {
            "pixel_values": torch.tensor(values, dtype=torch.float32).reshape(-1, 1),
            "image_grid_thw": torch.tensor([[1, 4, 4] for _ in images], dtype=torch.long),
        }


class _FakeVisionEncoder(nn.Module):
    spatial_merge_size = 2

    def __init__(self, hidden_size=8):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.hidden_size = hidden_size

    def forward(self, pixel_values, *, grid_thw, return_dict):
        features = [
            torch.full((4, self.hidden_size), float(value.item()), device=pixel_values.device)
            for value in pixel_values[:, 0]
        ]
        return SimpleNamespace(pooler_output=torch.cat(features, dim=0))


def _position_indexer():
    full_config = Qwen3_5Config(
        vision_config=Qwen3_5VisionConfig(spatial_merge_size=2),
        image_token_id=5,
        vision_start_token_id=6,
        vision_end_token_id=7,
    )
    return full_config, Qwen3_5VisionPositionIndexer(full_config)


class MossImageHQVisionTests(unittest.TestCase):
    def test_text_encoder_keeps_text_only_path_and_accepts_image_embeddings_with_mrope(self):
        config = Qwen3_5TextConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            layer_types=["full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 1],
            },
        )
        encoder = TextEncoder(config).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        inputs_embeds = encoder.embed_tokens(input_ids)
        explicit_text_positions = torch.arange(4).reshape(1, 4)
        with torch.inference_mode():
            text_only = encoder(input_ids)
            text_multimodal_api = encoder(
                input_ids,
                inputs_embeds=inputs_embeds,
                position_ids=explicit_text_positions,
            )
            mrope_positions = explicit_text_positions.unsqueeze(0).expand(3, -1, -1).clone()
            mrope_positions[1, 0, 2:] += 1
            multimodal = encoder(input_ids, inputs_embeds=inputs_embeds, position_ids=mrope_positions)
            image_input_ids = torch.tensor([[1, 5, 2, 3]])
            image_positions = torch.arange(4).reshape(1, 4).unsqueeze(0).expand(3, -1, -1)
            scattered = encoder(
                image_input_ids,
                position_ids=image_positions,
                image_features=torch.full((1, 16), 2.0),
                image_token_id=5,
            )
        torch.testing.assert_close(text_multimodal_api, text_only, rtol=0, atol=0)
        self.assertEqual(tuple(multimodal.shape), (1, 4, 16))
        self.assertTrue(torch.isfinite(multimodal).all())
        self.assertEqual(tuple(scattered.shape), (1, 4, 16))
        self.assertTrue(torch.isfinite(scattered).all())

    def test_nonpersistent_vision_rotary_buffers_are_rebuilt_after_meta_loading(self):
        with init_empty_weights(include_buffers=True):
            rotary = Qwen3_5VisionRotaryEmbedding(dim=8)
            holder = nn.Module()
            holder.rotary = rotary
        self.assertTrue(holder.rotary.inv_freq.is_meta)

        restore_vision_rotary_buffers(holder)

        expected = 1.0 / (10_000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
        self.assertEqual(holder.rotary.inv_freq.device.type, "cpu")
        self.assertEqual(holder.rotary._lock_dtype, torch.float32)
        torch.testing.assert_close(holder.rotary.inv_freq, expected)

    def test_prompt_scatter_preserves_order_and_builds_qwen_three_axis_positions(self):
        full_config, indexer = _position_indexer()
        tokenizer = _MultimodalTokenizer()
        encoder = _RecordingTextEncoder()
        grids = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
        features = (torch.full((4, 8), 11.0), torch.full((4, 8), 29.0))

        result = encode_prompt_with_vision(
            encoder,
            tokenizer,
            "replace the item in image 1 with the object from image 2",
            features,
            grids,
            full_config,
            {"image": 5, "vision_start": 6, "vision_end": 7},
            indexer,
        )

        self.assertEqual(tuple(result.shape), tuple(encoder.calls[0][1].shape))
        input_ids, inputs_embeds, position_ids = encoder.calls[0]
        image_positions = input_ids.eq(5)[0]
        image_indices = torch.where(image_positions)[0]
        self.assertEqual(int(image_positions.sum()), 8)
        self.assertTrue(torch.all(inputs_embeds[0, image_indices[:4]] == 11.0))
        self.assertTrue(torch.all(inputs_embeds[0, image_indices[4:]] == 29.0))
        self.assertEqual(tuple(position_ids.shape), (3, 1, input_ids.shape[1]))
        self.assertGreater(torch.unique(position_ids[1, 0, image_indices[:4]]).numel(), 1)
        contents = tokenizer.rendered_messages[0][0][0]["content"]
        self.assertIn("Picture 1: <|vision_start|>", contents)
        self.assertIn("Picture 2: <|vision_start|>", contents)

    def test_both_cfg_branches_receive_references_and_cache_is_scoped_to_ordered_pixels(self):
        full_config, indexer = _position_indexer()
        text_encoder = _RecordingTextEncoder()
        processor = _FakeImageProcessor()
        vision_encoder = _FakeVisionEncoder()
        tokenizer = _MultimodalTokenizer()
        pipeline = MOSSImagePipeline.__new__(MOSSImagePipeline)
        pipeline.hq_edit = True
        pipeline.text_encoder = text_encoder
        pipeline.text_encoder_cache = TextEncoderCache()
        pipeline.tokenizer = tokenizer
        pipeline.vision_encoder = vision_encoder
        pipeline.image_processor = processor
        pipeline.vision_config = full_config
        pipeline.vision_token_ids = {"image": 5, "vision_start": 6, "vision_end": 7}
        pipeline.vision_position_indexer = indexer
        pipeline.vision_max_pixels = 512 * 512

        refs = [Image.new("RGB", (16, 16), (13, 0, 0)), Image.new("RGB", (16, 16), (37, 0, 0))]
        prompts = ["", "replace image 1 with image 2"]
        outputs = pipeline._encode_text_contexts(prompts, refs, "cpu")
        self.assertEqual(len(outputs), 2)
        self.assertEqual(len(text_encoder.calls), 2)
        for input_ids, inputs_embeds, positions in text_encoder.calls:
            image_positions = input_ids.eq(5)[0]
            image_indices = torch.where(image_positions)[0]
            self.assertEqual(int(image_positions.sum()), 8)
            self.assertEqual(tuple(positions.shape), (3, 1, input_ids.shape[1]))
            self.assertTrue(torch.all(inputs_embeds[0, image_indices[:4]] == 13.0))
            self.assertTrue(torch.all(inputs_embeds[0, image_indices[4:]] == 37.0))
        self.assertEqual(processor.sizes, [{"shortest_edge": 56 * 56, "longest_edge": 512 * 512}])

        # Same refs reuse prompt states; reversing the refs must encode both CFG arms again.
        pipeline._encode_text_contexts(prompts, refs, "cpu")
        self.assertEqual(len(text_encoder.calls), 2)
        pipeline._encode_text_contexts(prompts, list(reversed(refs)), "cpu")
        self.assertEqual(len(text_encoder.calls), 4)
        for input_ids, inputs_embeds, _ in text_encoder.calls[2:]:
            image_positions = input_ids.eq(5)[0]
            image_indices = torch.where(image_positions)[0]
            self.assertTrue(torch.all(inputs_embeds[0, image_indices[:4]] == 37.0))
            self.assertTrue(torch.all(inputs_embeds[0, image_indices[4:]] == 13.0))

    def test_cache_key_covers_prompt_pixels_order_mode_and_vision_layout(self):
        first = Image.new("RGB", (2, 3), (1, 2, 3))
        second = Image.new("RGB", (2, 3), (4, 5, 6))
        baseline = vision_prompt_cache_key("edit", [first, second], 512 * 512)
        self.assertNotEqual(baseline, vision_prompt_cache_key("edit", [second, first], 512 * 512))
        self.assertNotEqual(baseline, vision_prompt_cache_key("other", [first, second], 512 * 512))
        self.assertNotEqual(baseline, vision_prompt_cache_key("edit", [first, second], 384 * 384))
        second.format = "PNG"
        self.assertNotEqual(baseline, vision_prompt_cache_key("edit", [first, second], 512 * 512))

    def test_missing_vision_assets_fail_clearly_without_text_only_fallback(self):
        with patch("models.moss_image.hq_vision.fl.locate_file", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "Download the HQ vision assets"):
                load_hq_vision_components({
                    "moss_edit_vision_folder": "missing_hq_vision",
                    "moss_edit_vision_file": "nested/vision.safetensors",
                })

    def test_loader_materializes_nonpersistent_rotary_before_offload_state_validation(self):
        full_config, _ = _position_indexer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            processor_path = root / "preprocessor_config.json"
            checkpoint_path = root / "nested" / "vision.safetensors"
            checkpoint_path.parent.mkdir()
            config_path.touch()
            processor_path.touch()
            checkpoint_path.touch()
            paths = {
                "config.json": str(config_path),
                "preprocessor_config.json": str(processor_path),
                "nested/vision.safetensors": str(checkpoint_path),
            }

            def fake_vision_factory(_config):
                model = nn.Module()
                model.rotary = Qwen3_5VisionRotaryEmbedding(dim=8)
                return model

            def validate_weights_load(model, _paths, **kwargs):
                self.assertFalse(model.rotary.inv_freq.is_meta)
                self.assertEqual(model.rotary.inv_freq.device.type, "cpu")
                self.assertEqual(model.rotary._lock_dtype, torch.float32)
                self.assertEqual(kwargs["modelPrefix"], "model.visual")

            with (
                patch.object(
                    hq_vision_module,
                    "_required_asset",
                    side_effect=lambda _folder, relative_path, _label: paths[relative_path],
                ),
                patch.object(Qwen3_5Config, "from_json_file", return_value=full_config),
                patch.object(hq_vision_module, "Qwen3_5VisionModel", side_effect=fake_vision_factory),
                patch.object(hq_vision_module.offload, "load_model_data", side_effect=validate_weights_load),
                patch.object(hq_vision_module, "_load_qwen35_image_processor", return_value=object()),
            ):
                vision_encoder, _, loaded_config, token_ids = load_hq_vision_components({
                    "moss_edit_vision_folder": str(root),
                    "moss_edit_vision_file": "nested/vision.safetensors",
                })

        self.assertFalse(vision_encoder.rotary.inv_freq.is_meta)
        self.assertIs(loaded_config, full_config)
        self.assertEqual(token_ids["image"], full_config.image_token_id)


if __name__ == "__main__":
    unittest.main()
