"""Qwen3-VL layer-50 conditioning for MiniMax H3."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2VLImageProcessorFast

from models.ideogram4.qwen3_vl_configuration import Qwen3VLConfig, register_qwen3_vl_config
from models.ideogram4.qwen3_vl_transformers import Qwen3VLModel, Qwen3VLTextModel, Qwen3VLVisionModel
from models.krea2.krea2_main import Krea2Qwen3VLProcessor


VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IMAGE_TOKEN_ID = 151655
TEXT_ENCODER_LAYERS = 50


class MiniMaxH3PreScaledLinear(nn.Linear):
    """AWQ/NVFP4 linear with the checkpoint's input smoothing scale."""

    def __init__(self, in_features: int, out_features: int, bias: bool, dtype: torch.dtype):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.register_buffer("pre_quant_scale", torch.empty(in_features, dtype=dtype), persistent=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.pre_quant_scale.to(device=input.device, dtype=input.dtype)
        return F.linear(input * scale, self.weight, self.bias)


class MiniMaxH3Qwen3VL(nn.Module):
    """Checkpoint-shaped Qwen3-VL wrapper.

    The consumer checkpoint uses the top-level prefixes ``model`` and
    ``visual`` and ends after decoder layer 50.  H3 consumes that layer's
    unnormalized output, so the absent final norm is intentionally replaced by
    an identity module.
    """

    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.model = Qwen3VLTextModel(config.text_config)
        self.model.norm = nn.Identity()
        for layer in self.model.layers:
            down = layer.mlp.down_proj
            layer.mlp.down_proj = MiniMaxH3PreScaledLinear(
                down.in_features,
                down.out_features,
                down.bias is not None,
                down.weight.dtype,
            )
            out = layer.self_attn.o_proj
            layer.self_attn.o_proj = MiniMaxH3PreScaledLinear(
                out.in_features,
                out.out_features,
                out.bias is not None,
                out.weight.dtype,
            )

    get_rope_index = Qwen3VLModel.get_rope_index


def load_h3_qwen_config(config_path: str) -> Qwen3VLConfig:
    register_qwen3_vl_config()
    config = Qwen3VLConfig.from_json_file(config_path)
    config.text_config.num_hidden_layers = TEXT_ENCODER_LAYERS
    return config


def build_h3_processor(config_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(config_dir, trust_remote_code=False)
    image_processor = Qwen2VLImageProcessorFast.from_pretrained(config_dir)
    return tokenizer, Krea2Qwen3VLProcessor(image_processor, tokenizer)


def _tag_vision_spans(input_ids: torch.Tensor) -> torch.Tensor:
    """Return H3 modality tags, including the vision boundary tokens."""

    ids = input_ids[0].tolist()
    tags = torch.ones(len(ids), dtype=torch.long)
    start = None
    for index, token in enumerate(ids):
        if token == VISION_START_TOKEN_ID:
            start = index
        if token == VISION_END_TOKEN_ID and start is not None:
            tags[start : index + 1] = 0
            start = None
    if start is not None:
        tags[start:] = 0
    return tags


class MiniMaxH3Conditioner(nn.Module):
    def __init__(self, qwen: MiniMaxH3Qwen3VL, tokenizer, processor, max_text_tokens: int = 512):
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_text_tokens = max_text_tokens
        self._interrupt = False

    @property
    def language_model(self):
        return self.qwen.model

    @property
    def visual(self):
        return self.qwen.visual

    def _plain_inputs(self, prompt: str, device: torch.device):
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).bool()
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(~attention_mask, 1)
        return input_ids, attention_mask, position_ids, None

    def _vision_inputs(self, prompt: str, images: list, device: torch.device):
        presentation = "".join(
            f"<Picture {index + 1}>: <|vision_start|><|image_pad|><|vision_end|>"
            for index in range(len(images))
        ) + prompt
        encoded = self.processor(
            text=[presentation],
            images=images,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_text_tokens + 4096,
            return_tensors="pt",
        ).to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()
        return input_ids, attention_mask, None, encoded

    @torch.inference_mode()
    def forward(self, prompt: str, device: torch.device, images: list | None = None):
        self.qwen.model._interrupt = self._interrupt
        self.qwen.visual._interrupt = self._interrupt
        if self._interrupt:
            return None, None
        if images:
            input_ids, attention_mask, position_ids, processor_inputs = self._vision_inputs(prompt, images, device)
            grid = processor_inputs["image_grid_thw"]
            pixels = processor_inputs["pixel_values"].to(device=device, dtype=torch.float32)
            image_embeds, deepstack = self.qwen.visual(pixels, grid_thw=grid)
            if image_embeds is None or self._interrupt:
                return None, None
            inputs_embeds = self.qwen.model.embed_tokens(input_ids)
            visual_mask = input_ids == IMAGE_TOKEN_ID
            inputs_embeds = inputs_embeds.masked_scatter(
                visual_mask.unsqueeze(-1).expand_as(inputs_embeds),
                image_embeds.to(inputs_embeds.dtype),
            )
            position_ids, _ = self.qwen.get_rope_index(
                input_ids,
                image_grid_thw=grid,
                attention_mask=attention_mask,
            )
        else:
            input_ids, attention_mask, position_ids, _ = self._plain_inputs(prompt, device)
            inputs_embeds = visual_mask = deepstack = None

        outputs = self.qwen.model(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack,
            return_mid_results_layers=[TEXT_ENCODER_LAYERS - 1],
        )
        if outputs.last_hidden_state is None or not outputs.mid_results:
            return None, None
        # The layer snapshot is taken before the (absent) final norm.
        embeddings = outputs.mid_results[0]
        tags = _tag_vision_spans(input_ids)
        return embeddings, tags
