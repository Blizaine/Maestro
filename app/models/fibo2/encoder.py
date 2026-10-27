# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
from torch import nn
from transformers import Qwen2VLProcessor
from transformers.processing_utils import ProcessorMixin

from models.ideogram4.qwen3_vl_transformers import (
    Qwen3VLModel, Qwen3VLTextModel, Qwen3VLVisionModel,
    Qwen3VLTextAttention, _take_tensor,
)
from shared import attention_kit
from . import ops
from .attention import attend_projected
from .progress import GenerationAborted, text_encoding_progress
from .runtime import runtime

LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))
PROMPT_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class FiboTextAttention(Qwen3VLTextAttention):
    """Conditioning-only Qwen attention: no autoregressive KV cache."""

    def forward(self, hidden_states, position_embeddings, attention_mask, **kwargs):
        hidden_states = _take_tensor(hidden_states)
        batch, length, _ = hidden_states.shape
        cosine, sine = position_embeddings
        query = self.q_proj(hidden_states).view(
            batch, length, self.config.num_attention_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch, length, self.config.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch, length, self.config.num_key_value_heads, self.head_dim
        )
        query = ops.paired_rope_(self.q_norm(query), cosine[:, :, None], sine[:, :, None])
        key = ops.paired_rope_(self.k_norm(key), cosine[:, :, None], sine[:, :, None])
        attended = attend_projected(
            query, key, value, attention_mask=attention_mask,
            causal=attention_mask is None, force_attention="sdpa",
            # Local Qwen3-VL turns tokenizer masks into [batch, query, 1, key]
            # before calling the layer; pay_attention uses this layout too.
            attention_mask_layout="maestro" if attention_mask is not None and attention_mask.ndim == 4 else "standard",
        )
        return self.o_proj(attended.flatten(2)), None


class FiboVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        hidden_states = _take_tensor(hidden_states)
        cosine, sine = position_embeddings
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        output = torch.empty_like(hidden_states)
        cursor = 0
        for length in lengths:
            def norm_rope(q, k, group):
                for tensor in (q, k):
                    if tensor is not None:
                        def rows(part, first, last):
                            value = part.float()
                            return ops.paired_rope_(
                                value, cosine[None, cursor + first:cursor + last, None].float(),
                                sine[None, cursor + first:cursor + last, None].float(),
                            ).to(part.dtype)
                        ops.rows_(tensor, rows)
            value = attention_kit.qkv_attention(
                [hidden_states[cursor:cursor + length]], self.q_proj, self.k_proj, self.v_proj,
                self.num_heads, self.head_dim, norm_rope,
            )
            output[cursor:cursor + length].copy_(value.reshape(length, -1))
            cursor += length
        del hidden_states
        return self.proj(output)


class FiboProcessor(Qwen2VLProcessor):
    attributes = ["image_processor", "tokenizer"]

    def __init__(self, image_processor, tokenizer):
        self.image_token, self.video_token = "<|image_pad|>", "<|video_pad|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        ProcessorMixin.__init__(
            self, image_processor, tokenizer,
            chat_template=getattr(tokenizer, "chat_template", None),
        )


class FiboEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel(config.text_config)
        for index, layer in enumerate(self.language_model.layers):
            layer.self_attn = FiboTextAttention(config.text_config, index)
        for block in self.visual.blocks:
            block.attn = FiboVisionAttention(config.vision_config)
        self.visual.rotary_pos_emb._lock_dtype = torch.float32
        self.language_model.rotary_emb._lock_dtype = torch.float32
        self.split_linear_modules_map = {
            "qkv": {"mapped_modules": ["q_proj", "k_proj", "v_proj"], "split_sizes": [config.vision_config.hidden_size] * 3},
        }
        self._interrupt = False

    get_rope_index = Qwen3VLModel.get_rope_index

    def forward(self, inputs):
        self.language_model._interrupt = self.visual._interrupt = self._interrupt
        device = runtime.execution_device(self)
        inputs = inputs.to(device)
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        embeddings = visual_mask = deepstack = None
        if "pixel_values" in inputs:
            vision, deepstack = self.visual(inputs["pixel_values"].to(torch.bfloat16), grid_thw=inputs["image_grid_thw"])
            if self._interrupt:
                raise GenerationAborted
            embeddings = self.language_model.embed_tokens(ids)
            visual_mask = ids == self.config.image_token_id
            embeddings = embeddings.masked_scatter(visual_mask.unsqueeze(-1).expand_as(embeddings), vision.to(embeddings.dtype))
            del vision
            positions, _ = self.get_rope_index(ids, image_grid_thw=inputs["image_grid_thw"], attention_mask=mask)
        else:
            positions = mask.long().cumsum(-1) - 1
        selected = sorted({layer for bundle in LAYERS for layer in bundle})
        with text_encoding_progress(self.language_model.layers):
            output = self.language_model(input_ids=ids if embeddings is None else None, inputs_embeds=embeddings,
                                         attention_mask=mask, position_ids=positions, use_cache=False,
                                         visual_pos_masks=visual_mask, deepstack_visual_embeds=deepstack,
                                         return_mid_results_layers=[layer - 1 for layer in selected])
        if self._interrupt or output.last_hidden_state is None:
            raise GenerationAborted
        normalized = {}
        for layer, states in zip(selected, output.mid_results):
            value = states.float()
            normalized[layer] = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
        del output, embeddings, deepstack, inputs
        return torch.stack([torch.cat([normalized[layer] for layer in bundle], -1) for bundle in LAYERS], 1)
