# Copyright 2026 The MiniMax and Hugging Face teams.
# Copyright 2026 Maestro contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MMGP-native MiniMax H3 transformer for the compact consumer checkpoints.

The released Comfy-Org checkpoints replace H3's large timestep MLP and AdaLN
inputs with a sampled eight-dimensional curve.  This implementation keeps the
checkpoint's fused QKV and SwiGLU projections intact so Maestro's FP8 loader can
stream them without first expanding or dequantizing the 21 GB transformer.

Packing, modality tags, schedules, and rotary coordinates follow the official
Diffusers MiniMax H3 implementation pinned in ``UPSTREAM.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITY_VIDEO = 0
MODALITY_TEXT = 1
MODALITY_AUDIO = 2
MODALITY_COUNT = 3


@dataclass
class MiniMaxH3TransformerOutput:
    sample: torch.Tensor
    audio_sample: torch.Tensor


def _weight_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    weight = getattr(module, "weight", None)
    dtype = getattr(weight, "dtype", None)
    if dtype is None or dtype == torch.uint8:
        return fallback
    return dtype


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply split-half RoPE to the leading rotary channels."""

    rotary_dim = cos.shape[-1]
    rotary, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    cos = cos.to(dtype=x.dtype, device=x.device)[None, :, None]
    sin = sin.to(dtype=x.dtype, device=x.device)[None, :, None]
    rotary = rotary * cos + rotated * sin
    return torch.cat((rotary, passthrough), dim=-1)


class MiniMaxH3RotaryEmbedding(nn.Module):
    def __init__(self, freq_dim: int = 16, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * freq_dim, 2, dtype=torch.float32) / (2 * freq_dim)))
        # Consumer checkpoints include this tensor, so keep it persistent.
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(device=self.inv_freq.device, dtype=torch.float32)
        angles = positions.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        temporal, vertical, horizontal = angles.unbind(dim=1)
        angles = torch.cat((temporal, vertical, horizontal), dim=-1)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()


class MiniMaxH3Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, head_dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = nn.Linear(hidden_size, inner * 3, bias=False, dtype=dtype)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps, dtype=dtype)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps, dtype=dtype)
        self.out_proj = nn.Linear(inner, hidden_size, bias=False, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self.q_norm(query.view(batch, length, self.heads, self.head_dim))
        key = self.k_norm(key.view(batch, length, self.heads, self.head_dim))
        value = value.view(batch, length, self.heads, self.head_dim)
        if rotary is not None:
            query = _apply_rope(query, *rotary)
            key = _apply_rope(key, *rotary)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if attention_mask is not None:
            attention_mask = attention_mask[None, None].to(device=query.device)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, self.heads * self.head_dim)
        return self.out_proj(attended)


class MiniMaxH3MLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int, dtype: torch.dtype):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_dim * 2, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(ffn_dim, hidden_size, bias=False, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The released H3/Comfy checkpoint stores the fused projection as
        # [gate, value].  Keeping that native order avoids a 14k x 10k tensor
        # rewrite while loading the quantized transformer.
        gate, value = self.fc1(hidden_states).chunk(2, dim=-1)
        return self.fc2(value * F.silu(gate))


class MiniMaxH3AdaLNProjection(nn.Module):
    def __init__(
        self,
        curve_dim: int,
        hidden_size: int,
        outputs: int,
        modalities: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.outputs = outputs
        self.modalities = modalities
        self.linear = nn.Linear(curve_dim, outputs * modalities * hidden_size, bias=True, dtype=dtype)

    def forward(self, curve: torch.Tensor) -> tuple[torch.Tensor, ...]:
        dtype = _weight_dtype(self.linear, curve.dtype)
        projected = self.linear(curve.to(dtype=dtype))
        projected = projected.view(curve.shape[0] * self.modalities, self.outputs * self.hidden_size)
        return projected.chunk(self.outputs, dim=-1)


class MiniMaxH3RefinerBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, head_dim: int, ffn_dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.norm2 = nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.attn = MiniMaxH3Attention(hidden_size, heads, head_dim, eps, dtype)
        self.mlp = MiniMaxH3MLP(hidden_size, ffn_dim, dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        layers: int,
        hidden_size: int,
        heads: int,
        head_dim: int,
        ffn_dim: int,
        eps: float,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MiniMaxH3RefinerBlock(hidden_size, heads, head_dim, ffn_dim, eps, dtype) for _ in range(layers)]
        )
        self.final_norm = nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class MiniMaxH3Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        head_dim: int,
        ffn_dim: int,
        curve_dim: int,
        eps: float,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.norm2 = nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.attn = MiniMaxH3Attention(hidden_size, heads, head_dim, eps, dtype)
        self.mlp = MiniMaxH3MLP(hidden_size, ffn_dim, dtype)
        self.adaln_proj = MiniMaxH3AdaLNProjection(curve_dim, hidden_size, 6, MODALITY_COUNT, torch.float16)

    @staticmethod
    def _modulate(
        hidden_states: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        shift = shift.index_select(0, indices).to(hidden_states.dtype)
        scale = scale.index_select(0, indices).to(hidden_states.dtype)
        return hidden_states * (1.0 + scale) + shift

    def forward(
        self,
        hidden_states: torch.Tensor,
        curve: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(curve)
        normed = self._modulate(self.norm1(hidden_states), shift_attn, scale_attn, adaln_indices)
        hidden_states = hidden_states + gate_attn.index_select(0, adaln_indices).to(hidden_states.dtype) * self.attn(
            normed, rotary, attention_mask
        )
        normed = self._modulate(self.norm2(hidden_states), shift_mlp, scale_mlp, adaln_indices)
        return hidden_states + gate_mlp.index_select(0, adaln_indices).to(hidden_states.dtype) * self.mlp(normed)


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, curve_dim: int, video_dim: int, audio_dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.adaln_proj = MiniMaxH3AdaLNProjection(curve_dim, hidden_size, 2, 1, torch.float16)
        self.video_out = nn.Linear(hidden_size, video_dim, bias=True, dtype=torch.float32)
        self.audio_out = nn.Linear(hidden_size, audio_dim, bias=True, dtype=torch.float32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        curve: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.adaln_proj(curve)
        normed = self.norm(hidden_states)
        shift = shift.index_select(0, timestep_indices).to(normed.dtype)
        scale = scale.index_select(0, timestep_indices).to(normed.dtype)
        return normed * (1.0 + scale) + shift


class MiniMaxH3Transformer(nn.Module):
    """Compact-curve MiniMax H3 FL2VA transformer."""

    def __init__(
        self,
        hidden_size: int = 5376,
        num_layers: int = 50,
        token_refiner_layers: int = 2,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        ffn_dim: int = 14336,
        video_channels: int = 24,
        audio_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        curve_grid: int = 1025,
        curve_dim: int = 8,
        rope_freq_dim: int = 16,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        video_patch_dim = video_channels * math.prod(patch_size)
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_layers=num_layers,
            patch_size=patch_size,
            in_channels=video_channels,
            audio_in_channels=audio_channels,
            text_dim=text_dim,
            curve_grid=curve_grid,
            curve_dim=curve_dim,
        )
        self.video_patch_proj = nn.Linear(video_patch_dim, hidden_size, bias=True, dtype=torch.float32)
        self.audio_patch_proj = nn.Linear(audio_channels, hidden_size, bias=True, dtype=torch.float32)
        self.condition_proj = nn.Linear(text_dim, hidden_size, bias=True, dtype=dtype)
        self.register_buffer("adaln_t_table", torch.empty(curve_grid, curve_dim, dtype=torch.float32), persistent=True)
        self.rope = MiniMaxH3RotaryEmbedding(rope_freq_dim)
        self.token_refiner = MiniMaxH3TokenRefiner(
            token_refiner_layers,
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            ffn_dim,
            eps,
            dtype,
        )
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3Block(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_dim,
                    curve_dim,
                    eps,
                    dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = MiniMaxH3FinalLayer(
            hidden_size,
            curve_dim,
            video_patch_dim,
            audio_channels,
            eps,
            dtype,
        )
        self._interrupt = False

    def _curve_at(self, timestep: torch.Tensor, device: torch.device) -> torch.Tensor:
        table = self.adaln_t_table.to(device=device, dtype=torch.float32)
        position = timestep.to(device=device, dtype=torch.float32).clamp_(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp_(max=table.shape[0] - 2)
        fraction = (position - lower).unsqueeze(-1)
        return torch.lerp(table.index_select(0, lower), table.index_select(0, lower + 1), fraction)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        return_dict: bool = True,
        **_kwargs,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor] | None:
        if self._interrupt:
            return None
        if hidden_states.shape[0] != 1:
            raise ValueError("MiniMax H3 currently supports batch size 1.")
        sequence_length = position_ids.shape[0]
        if position_ids.shape != (sequence_length, 3):
            raise ValueError("MiniMax H3 position_ids must have shape [sequence, 3].")
        device = hidden_states.device
        video_indices = video_indices.to(device=device, dtype=torch.long)
        audio_indices = audio_indices.to(device=device, dtype=torch.long)
        text_indices = text_indices.to(device=device, dtype=torch.long)
        timestep_indices = timestep_indices.to(device=device, dtype=torch.long)
        token_tags = token_tags.to(device=device, dtype=torch.long)

        video_dtype = _weight_dtype(self.video_patch_proj, torch.float32)
        audio_dtype = _weight_dtype(self.audio_patch_proj, torch.float32)
        text_dtype = _weight_dtype(self.condition_proj, torch.bfloat16)
        video_embeds = self.video_patch_proj(hidden_states.to(dtype=video_dtype))
        audio_embeds = self.audio_patch_proj(audio_hidden_states.to(dtype=audio_dtype))
        text_embeds = self.condition_proj(encoder_hidden_states.to(dtype=text_dtype))
        text_embeds = self.token_refiner(text_embeds)

        packed = text_embeds.new_zeros((1, sequence_length, text_embeds.shape[-1]))
        packed.index_copy_(1, text_indices, text_embeds)
        packed.index_copy_(1, video_indices, video_embeds.to(packed.dtype))
        packed.index_copy_(1, audio_indices, audio_embeds.to(packed.dtype))

        curve = self._curve_at(timestep, device)
        adaln_indices = timestep_indices * MODALITY_COUNT + token_tags.clamp_min(0)
        rotary = self.rope(position_ids.to(device))
        attention_mask = None
        padding = token_tags < 0
        if bool(padding.any()):
            attention_mask = padding[:, None] == padding[None, :]

        for block in self.blocks:
            if self._interrupt:
                return None
            packed = block(packed, curve, adaln_indices, rotary, attention_mask)

        packed = self.final_layer(packed, curve, timestep_indices)
        video_activations = packed.index_select(1, video_indices).to(torch.float32)
        audio_activations = packed.index_select(1, audio_indices).to(torch.float32)
        video_output = self.final_layer.video_out(video_activations)
        audio_output = self.final_layer.audio_out(audio_activations)
        if not return_dict:
            return video_output, audio_output
        return MiniMaxH3TransformerOutput(video_output, audio_output)
