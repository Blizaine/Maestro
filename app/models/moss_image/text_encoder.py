# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
from torch import nn
from shared import attention_kit
from shared.attention import pay_attention
from shared.prompt_enhancer.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig as Qwen3_5TextConfig
from shared.prompt_enhancer.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding
from . import ops
from .progress import GenerationAborted, text_encoding_progress
from .runtime import runtime


def _expand_grouped_kv_for_sdpa(qkv):
    """Expand each K/V head across its query group for legacy SDPA callers."""
    query, key, value = qkv
    query_heads, key_heads, value_heads = query.shape[-2], key.shape[-2], value.shape[-2]
    if key_heads != value_heads:
        raise ValueError(
            f"MOSS attention received different K and V head counts ({key_heads} and {value_heads})."
        )
    if query_heads == key_heads:
        return
    if key_heads <= 0 or query_heads % key_heads:
        raise ValueError(
            f"MOSS attention requires query heads ({query_heads}) to be divisible by "
            f"key/value heads ({key_heads})."
        )
    repeat_count = query_heads // key_heads
    qkv[1] = key.repeat_interleave(repeat_count, dim=-2)
    qkv[2] = value.repeat_interleave(repeat_count, dim=-2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        width = self.heads * self.head_dim
        self.q_proj, self.gate_proj = (nn.Linear(config.hidden_size, width, bias=False) for _ in range(2))
        self.k_proj, self.v_proj = (nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False) for _ in range(2))
        self.o_proj = nn.Linear(width, config.hidden_size, bias=False)
        self.q_norm, self.k_norm = (Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps) for _ in range(2))

    def forward(self, hidden_states, position_embeddings, attention_mask, **kwargs):
        gate = self.gate_proj(hidden_states).sigmoid()
        cosine, sine = position_embeddings
        rotary_dim = cosine.shape[-1]
        def normalize(q, k, group):
            for tensor, norm in ((q, self.q_norm), (k, self.k_norm)):
                if tensor is not None:
                    def rows(part, first, last):
                        part = norm(part)
                        rotated, tail = part[..., :rotary_dim], part[..., rotary_dim:]
                        a, b = rotated.chunk(2, -1)
                        rotated = rotated * cosine[:, first:last, None] + torch.cat((-b, a), -1) * sine[:, first:last, None]
                        return torch.cat((rotated, tail), -1)
                    ops.rows_(tensor, rows)
        inputs, hidden_states = [hidden_states], None
        def attention(qkv):
            _expand_grouped_kv_for_sdpa(qkv)
            return pay_attention(
                qkv,
                attention_mask=attention_mask,
                force_attention="sdpa",
                recycle_q=True,
            )
        result = attention_kit.qkv_attention(
            inputs,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.heads,
            self.head_dim,
            normalize,
            kv_heads=self.kv_heads,
            attention_fn=attention,
        )
        return self.o_proj(result.flatten(2) * gate), None


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        config._prompt_enhancer_safe_legacy = True
        self._interrupt = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        for layer in self.layers:
            if layer.layer_type == 'full_attention':
                layer.self_attn = Attention(config)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        width = config.num_attention_heads * config.head_dim
        self.split_linear_modules_map = {'q_gate_proj': {'mapped_modules': ['q_proj', 'gate_proj'], 'split_sizes': [width, width]}}

    def forward(self, input_ids, inputs_embeds=None, position_ids=None, image_features=None, image_token_id=None):
        """Encode text ids, or pre-embedded multimodal tokens with explicit RoPE ids."""
        if inputs_embeds is not None and image_features is not None:
            raise ValueError("MOSS Image text encoder accepts either inputs_embeds or image_features, not both.")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.ndim != 3 or tuple(hidden.shape[:2]) != tuple(input_ids.shape):
            raise ValueError("MOSS Image text encoder inputs_embeds must match input_ids batch and sequence dimensions.")
        if image_features is not None:
            if image_token_id is None:
                raise ValueError("MOSS Image text encoder needs image_token_id when image_features are supplied.")
            image_mask = input_ids.eq(int(image_token_id))
            if image_features.ndim != 2 or image_features.shape[1] != hidden.shape[2]:
                raise ValueError("MOSS Image text encoder image_features must have shape (image_tokens, hidden_size).")
            if int(image_mask.sum().item()) != image_features.shape[0]:
                raise ValueError("MOSS Image text encoder image features do not match image-token placeholders.")
            hidden = hidden.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(hidden),
                image_features.to(device=hidden.device, dtype=hidden.dtype).reshape(-1),
            )
        positions = position_ids
        if positions is None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        elif positions.ndim == 3 and tuple(positions.shape) != (3, input_ids.shape[0], input_ids.shape[1]):
            raise ValueError("MOSS Image text encoder position_ids must have shape (3, batch, sequence).")
        elif positions.ndim == 2 and tuple(positions.shape) != tuple(input_ids.shape):
            raise ValueError("MOSS Image text encoder position_ids must match input_ids batch and sequence dimensions.")
        embeddings = self.rotary_emb(hidden, positions)
        mask = torch.ones((1, input_ids.shape[1], 1, input_ids.shape[1]), device=hidden.device, dtype=torch.bool).transpose(1, 2).tril().transpose(1, 2)
        for layer in self.layers:
            hidden = layer(hidden, position_embeddings=embeddings, attention_mask=mask if layer.layer_type == 'full_attention' else None)
            if self._interrupt:
                raise GenerationAborted
        return self.norm(hidden)


def encode_prompt(encoder, tokenizer, prompt):
    formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(formatted, truncation=True, max_length=4096)
    ids = torch.tensor([inputs.input_ids], device=runtime.execution_device(encoder), dtype=torch.long)
    with text_encoding_progress(encoder.layers):
        return encoder(ids)
