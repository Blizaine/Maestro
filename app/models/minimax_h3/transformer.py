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

# A 10-second 480p H3 request contains well over 100,000 packed tokens.
# Projecting all of those tokens through the fused QKV and 2x-SwiGLU layers
# in one call creates 5-7 GB temporary tensors.  These projections are
# token-wise, so bounded chunks are mathematically equivalent and leave room
# for attention plus MMGP's streamed transformer blocks on consumer GPUs.
MINIMAX_H3_ACTIVATION_CHUNK_TOKENS = 8192


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


def _index_runs(indices: torch.Tensor) -> tuple[tuple[int, int, int], ...]:
    """Compress a token-to-curve map into contiguous broadcastable runs."""

    values, counts = torch.unique_consecutive(indices, return_counts=True)
    values = values.detach().cpu().tolist()
    counts = counts.detach().cpu().tolist()
    cursor = 0
    runs = []
    for value, count in zip(values, counts):
        end = cursor + int(count)
        runs.append((cursor, end, int(value)))
        cursor = end
    return tuple(runs)


def _modulate_by_runs(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    runs: tuple[tuple[int, int, int], ...],
) -> torch.Tensor:
    """Apply AdaLN without expanding shift and scale to every token."""

    # Inference owns this freshly-normalized tensor, so updating it in place
    # avoids another sequence x hidden-size allocation.  Keep an autograd-safe
    # path for the small numerical regression tests and downstream training.
    output = hidden_states if not torch.is_grad_enabled() else hidden_states.clone()
    for start, end, value in runs:
        row_scale = scale[value].to(device=output.device, dtype=output.dtype)
        row_shift = shift[value].to(device=output.device, dtype=output.dtype)
        output[:, start:end].mul_(1.0 + row_scale).add_(row_shift)
    return output


def _scale_by_runs(
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    runs: tuple[tuple[int, int, int], ...],
) -> torch.Tensor:
    """Apply a per-curve residual gate without a token-sized index_select."""

    output = hidden_states if not torch.is_grad_enabled() else hidden_states.clone()
    for start, end, value in runs:
        row_scale = scale[value].to(device=output.device, dtype=output.dtype)
        output[:, start:end].mul_(row_scale)
    return output


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
        chunk_size = max(1, int(MINIMAX_H3_ACTIVATION_CHUNK_TOKENS))
        if length <= chunk_size:
            qkv = self.qkv_proj(hidden_states)
            query, key, value = qkv.chunk(3, dim=-1)
            query = self.q_norm(query.view(batch, length, self.heads, self.head_dim))
            key = self.k_norm(key.view(batch, length, self.heads, self.head_dim))
            value = value.view(batch, length, self.heads, self.head_dim)
            if rotary is not None:
                query = _apply_rope(query, *rotary)
                key = _apply_rope(key, *rotary)
        else:
            # Keep only Q/K/V themselves resident.  The fused projection,
            # normalization, and RoPE temporaries are bounded to one chunk.
            shape = (batch, length, self.heads, self.head_dim)
            query = key = value = None
            for start in range(0, length, chunk_size):
                end = min(length, start + chunk_size)
                qkv = self.qkv_proj(hidden_states[:, start:end])
                q_chunk, k_chunk, v_chunk = qkv.chunk(3, dim=-1)
                chunk_length = end - start
                q_chunk = self.q_norm(
                    q_chunk.view(batch, chunk_length, self.heads, self.head_dim)
                )
                k_chunk = self.k_norm(
                    k_chunk.view(batch, chunk_length, self.heads, self.head_dim)
                )
                v_chunk = v_chunk.view(batch, chunk_length, self.heads, self.head_dim)
                if rotary is not None:
                    cos, sin = rotary
                    q_chunk = _apply_rope(q_chunk, cos[start:end], sin[start:end])
                    k_chunk = _apply_rope(k_chunk, cos[start:end], sin[start:end])
                if query is None:
                    query = torch.empty(shape, device=q_chunk.device, dtype=q_chunk.dtype)
                    key = torch.empty(shape, device=k_chunk.device, dtype=k_chunk.dtype)
                    value = torch.empty(shape, device=v_chunk.device, dtype=v_chunk.dtype)
                query[:, start:end].copy_(q_chunk)
                key[:, start:end].copy_(k_chunk)
                value[:, start:end].copy_(v_chunk)
            assert query is not None and key is not None and value is not None
            qkv = q_chunk = k_chunk = v_chunk = None
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
        query = key = value = qkv = None
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
        def project(rows: torch.Tensor) -> torch.Tensor:
            gate, value = self.fc1(rows).chunk(2, dim=-1)
            if not torch.is_grad_enabled():
                gate = F.silu(gate, inplace=True)
                gate.mul_(value)
                return self.fc2(gate)
            return self.fc2(value * F.silu(gate))

        length = hidden_states.shape[1]
        chunk_size = max(1, int(MINIMAX_H3_ACTIVATION_CHUNK_TOKENS))
        if length <= chunk_size:
            return project(hidden_states)

        output = torch.empty_like(hidden_states)
        for start in range(0, length, chunk_size):
            end = min(length, start + chunk_size)
            output[:, start:end].copy_(project(hidden_states[:, start:end]))
        return output


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
        # The compact curve checkpoint stores these projections in FP16, but
        # Comfy's reference curve path evaluates them in FP32.  Preserve the
        # compact storage dtype for MMGP and upcast only the tiny projection
        # while it is active; doing the multiply in FP16 compounds rounding
        # error coherently through all 50 transformer blocks.
        self.linear._lock_dtype = dtype

    def forward(self, curve: torch.Tensor) -> tuple[torch.Tensor, ...]:
        weight = self.linear.weight.to(device=curve.device, dtype=torch.float32)
        bias = self.linear.bias
        if bias is not None:
            bias = bias.to(device=curve.device, dtype=torch.float32)
        projected = F.linear(curve.to(dtype=torch.float32), weight, bias)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        curve: torch.Tensor,
        adaln_runs: tuple[tuple[int, int, int], ...],
        rotary: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(curve)
        normed = _modulate_by_runs(self.norm1(hidden_states), shift_attn, scale_attn, adaln_runs)
        attn_output = _scale_by_runs(self.attn(normed, rotary, attention_mask), gate_attn, adaln_runs)
        if not torch.is_grad_enabled():
            hidden_states.add_(attn_output)
        else:
            hidden_states = hidden_states + attn_output
        del normed, attn_output
        normed = _modulate_by_runs(self.norm2(hidden_states), shift_mlp, scale_mlp, adaln_runs)
        mlp_output = _scale_by_runs(self.mlp(normed), gate_mlp, adaln_runs)
        if not torch.is_grad_enabled():
            hidden_states.add_(mlp_output)
            return hidden_states
        return hidden_states + mlp_output


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, curve_dim: int, video_dim: int, audio_dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.adaln_proj = MiniMaxH3AdaLNProjection(curve_dim, hidden_size, 2, 1, torch.float16)
        self.video_out = nn.Linear(hidden_size, video_dim, bias=True, dtype=torch.float32)
        self.audio_out = nn.Linear(hidden_size, audio_dim, bias=True, dtype=torch.float32)
        # The output heads are the checkpoint's FP32 precision island.
        self.video_out._lock_dtype = torch.float32
        self.audio_out._lock_dtype = torch.float32

    def forward(
        self,
        hidden_states: torch.Tensor,
        curve: torch.Tensor,
        timestep_runs: tuple[tuple[int, int, int], ...],
    ) -> torch.Tensor:
        shift, scale = self.adaln_proj(curve)
        normed = self.norm(hidden_states)
        return _modulate_by_runs(normed, shift, scale, timestep_runs)


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
        # Input projections are also released and evaluated in FP32.
        self.video_patch_proj._lock_dtype = torch.float32
        self.audio_patch_proj._lock_dtype = torch.float32
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
        adaln_runs = _index_runs(adaln_indices)
        timestep_runs = _index_runs(timestep_indices)
        rotary = self.rope(position_ids.to(device))
        attention_mask = None
        padding = token_tags < 0
        if bool(padding.any()):
            attention_mask = padding[:, None] == padding[None, :]

        for block in self.blocks:
            if self._interrupt:
                return None
            packed = block(
                packed,
                curve,
                adaln_runs,
                rotary,
                attention_mask,
            )

        packed = self.final_layer(packed, curve, timestep_runs)
        video_activations = packed.index_select(1, video_indices).to(torch.float32)
        audio_activations = packed.index_select(1, audio_indices).to(torch.float32)
        video_output = self.final_layer.video_out(video_activations)
        audio_output = self.final_layer.audio_out(audio_activations)
        if not return_dict:
            return video_output, audio_output
        return MiniMaxH3TransformerOutput(video_output, audio_output)
