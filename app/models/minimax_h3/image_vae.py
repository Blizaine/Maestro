"""Native MiniMax H3 Image VAE.

The image checkpoint shares H3's causal 3D CNN encoder and per-channel latent
statistics. Its decoder is a 24-layer, 2D rotary ViT with a single 16x16 pixel
patch per latent. The decoder below is intentionally image-specific: it keeps
the attention global and exact while splitting queries and MLP rows into small
chunks so native 4K images do not require quadratic attention-score storage.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .video_vae import MiniMaxH3VideoEncoder3d


IMAGE_VAE_LATENTS_MEAN = (
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
)

IMAGE_VAE_LATENTS_STD = (
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
)

_PIXEL_MEAN = (0.485, 0.456, 0.406)
_PIXEL_STD = (0.229, 0.224, 0.225)
_IMAGE_ENCODE_SEED = 42
_IMAGE_SPATIAL_COMPRESSION = 16

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older supported torch builds
    SDPBackend = None
    sdpa_kernel = None


class MiniMaxH3ImageAttention(nn.Module):
    """Fused per-head QKV attention used by the image decoder."""

    def __init__(self, dim: int, heads: int, head_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.norm_q = nn.RMSNorm(head_dim, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps, elementwise_affine=False)
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.query_chunk_tokens = 512
        self.key_chunk_tokens = 4096

    @staticmethod
    def _normalize_and_rotate(
        values: torch.Tensor,
        norm: nn.RMSNorm,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        chunk_tokens: int,
        interrupt_check: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Normalize and apply split-half 2D RoPE without a full FP32 copy."""

        batch, sequence, heads, head_dim = values.shape
        rotary_dim = cos.shape[-1]
        rotated_values = torch.empty_like(values)
        for start in range(0, sequence, chunk_tokens):
            if interrupt_check is not None:
                interrupt_check()
            stop = min(start + chunk_tokens, sequence)
            part = norm(values[:, start:stop].float()).to(dtype=values.dtype)
            part_rope = part[..., :rotary_dim]
            first, second = part_rope.chunk(2, dim=-1)
            quarter_turn = torch.cat((-second, first), dim=-1)
            cos_part = cos[:, start:stop, None, :].to(dtype=values.dtype)
            sin_part = sin[:, start:stop, None, :].to(dtype=values.dtype)
            part = torch.cat(
                (part_rope * cos_part + quarter_turn * sin_part, part[..., rotary_dim:]),
                dim=-1,
            )
            rotated_values[:, start:stop] = part
        return rotated_values

    @staticmethod
    def _online_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_chunk_tokens: int,
        interrupt_check: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Exact dense softmax attention with bounded score storage.

        This streaming log-sum-exp fallback is used when the active PyTorch
        backend cannot run its fused CUDA attention kernel for a long key grid.
        It visits every key for every query and therefore preserves global
        attention semantics.
        """

        query_fp32 = query.float()
        scale = query.shape[-1] ** -0.5
        batch, heads, query_tokens, _ = query.shape
        output = torch.zeros_like(query_fp32)
        running_max = torch.full(
            (batch, heads, query_tokens, 1),
            -torch.inf,
            dtype=torch.float32,
            device=query.device,
        )
        running_sum = torch.zeros_like(running_max)

        for key_start in range(0, key.shape[-2], key_chunk_tokens):
            if interrupt_check is not None:
                interrupt_check()
            key_stop = min(key_start + key_chunk_tokens, key.shape[-2])
            key_part = key[:, :, key_start:key_stop].float()
            value_part = value[:, :, key_start:key_stop].float()
            scores = torch.matmul(query_fp32, key_part.transpose(-2, -1))
            scores.mul_(scale)
            part_max = scores.amax(dim=-1, keepdim=True)
            next_max = torch.maximum(running_max, part_max)

            previous_scale = torch.exp(running_max - next_max)
            scores.sub_(next_max).exp_()
            part_sum = scores.sum(dim=-1, keepdim=True)
            output.mul_(previous_scale)
            output.add_(torch.matmul(scores, value_part))
            running_sum.mul_(previous_scale)
            running_sum.add_(part_sum)
            running_max = next_max

        return (output / running_sum).to(dtype=query.dtype)

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        interrupt_check: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        sequence = query.shape[-2]
        if query.device.type == "cuda" and sdpa_kernel is not None and SDPBackend is not None:
            # Restrict CUDA dispatch to fused memory-efficient kernels. A math
            # fallback would allocate the full [Q,K] score matrix on a 4K grid.
            chunk_size = self.query_chunk_tokens if sequence > 8192 else sequence
            output = torch.empty_like(query)
            try:
                with sdpa_kernel(
                    backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
                ):
                    for start in range(0, sequence, chunk_size):
                        if interrupt_check is not None:
                            interrupt_check()
                        stop = min(start + chunk_size, sequence)
                        output[:, :, start:stop] = F.scaled_dot_product_attention(
                            query[:, :, start:stop], key, value, dropout_p=0.0, is_causal=False
                        )
                return output
            except (RuntimeError, NotImplementedError):
                # Long context limits and driver combinations vary. Retry with
                # an exact online softmax instead of allowing an OOM-prone math
                # backend to materialize the dense score matrix.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                chunk_size = self.query_chunk_tokens

        elif sequence <= 256:
            if interrupt_check is not None:
                interrupt_check()
            return F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        else:
            # CPU and older torch paths may have only the quadratic math
            # backend. Keep its query rows small, and switch to streaming online
            # attention for larger sequences.
            if sequence <= 1024:
                chunk_size = min(self.query_chunk_tokens, 128)
            else:
                chunk_size = self.query_chunk_tokens

        output = torch.empty_like(query)
        for start in range(0, sequence, chunk_size):
            if interrupt_check is not None:
                interrupt_check()
            stop = min(start + chunk_size, sequence)
            if query.device.type == "cuda" or sequence > 1024:
                part = self._online_attention(
                    query[:, :, start:stop],
                    key,
                    value,
                    key_chunk_tokens=self.key_chunk_tokens,
                    interrupt_check=interrupt_check,
                )
            else:
                part = F.scaled_dot_product_attention(
                    query[:, :, start:stop], key, value, dropout_p=0.0, is_causal=False
                )
            output[:, :, start:stop] = part
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        interrupt_check: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        if interrupt_check is not None:
            interrupt_check()
        batch, sequence, _ = hidden_states.shape
        packed = self.to_qkv(hidden_states).view(batch, sequence, self.heads, 3 * self.head_dim)
        query, key, value = packed.chunk(3, dim=-1)

        query = self._normalize_and_rotate(
            query,
            self.norm_q,
            cos,
            sin,
            chunk_tokens=self.query_chunk_tokens,
            interrupt_check=interrupt_check,
        ).transpose(1, 2)
        key = self._normalize_and_rotate(
            key,
            self.norm_k,
            cos,
            sin,
            chunk_tokens=self.query_chunk_tokens,
            interrupt_check=interrupt_check,
        ).transpose(1, 2)
        value = value.transpose(1, 2)

        attended = self._attend(
            query, key, value, interrupt_check=interrupt_check
        ).transpose(1, 2).reshape(batch, sequence, -1)
        return self.to_out(attended)


class MiniMaxH3ImageTransformerBlock(nn.Module):
    """One RMSNorm/SwiGLU residual block from the native image decoder."""

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        feed_forward_chunk_tokens: int = 4096,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = MiniMaxH3ImageAttention(dim, heads, head_dim, eps=eps)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = MiniMaxH3ImageFeedForward(dim, dim * ffn_mult)
        self.scale2 = nn.Parameter(torch.zeros(dim))
        self.feed_forward_chunk_tokens = feed_forward_chunk_tokens

    def _chunked_feed_forward(
        self,
        hidden_states: torch.Tensor,
        interrupt_check: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[1] <= self.feed_forward_chunk_tokens:
            if interrupt_check is not None:
                interrupt_check()
            return self.ff(hidden_states)

        parts = []
        for start in range(0, hidden_states.shape[1], self.feed_forward_chunk_tokens):
            if interrupt_check is not None:
                interrupt_check()
            parts.append(
                self.ff(hidden_states[:, start : start + self.feed_forward_chunk_tokens])
            )
        return torch.cat(parts, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        interrupt_check: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        if interrupt_check is not None:
            interrupt_check()
        normalized = self.norm1(hidden_states.float()).to(dtype=hidden_states.dtype)
        hidden_states = hidden_states + self.attn(
            normalized, cos, sin, interrupt_check=interrupt_check
        ) * self.scale1
        normalized = self.norm2(hidden_states.float()).to(dtype=hidden_states.dtype)
        hidden_states = hidden_states + self._chunked_feed_forward(
            normalized, interrupt_check=interrupt_check
        ) * self.scale2
        return hidden_states


class MiniMaxH3ImageFeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inner_dim * 2, bias=True)
        self.w2 = nn.Linear(inner_dim, dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, value = self.w1(hidden_states).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * value)


class MiniMaxH3ImageViTDecoder(nn.Module):
    """Single-frame, global-attention H3 image decoder."""

    def __init__(
        self,
        *,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        num_layers: int = 24,
        heads: int = 16,
        head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.5,
        norm_eps: float = 1e-5,
        feed_forward_chunk_tokens: int = 4096,
        attention_query_chunk_tokens: int = 512,
        attention_key_chunk_tokens: int = 4096,
    ) -> None:
        super().__init__()
        dim = heads * head_dim
        rotary_dim = int(head_dim * rope_dim_ratio)
        if rotary_dim % 4:
            raise ValueError("The H3 image 2D rotary dimension must be divisible by four.")
        if patch_size <= 0 or num_layers <= 0 or num_register_tokens < 0:
            raise ValueError("Image VAE decoder patch/layer/register settings must be positive.")

        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.rope_theta = float(rope_theta)
        self.rotary_dim = rotary_dim
        self.head_dim = head_dim
        self.heads = heads
        self.ffn_mult = ffn_mult
        # Keep the checkpoint's native `decoder.x_embedder` name. The H3 video
        # converter renames this to `proj_in`, but the image adapter does not.
        self.x_embedder = nn.Linear(in_channels, dim, bias=True)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3ImageTransformerBlock(
                    dim,
                    heads,
                    head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                    feed_forward_chunk_tokens=feed_forward_chunk_tokens,
                )
                for _ in range(num_layers)
            ]
        )
        for block in self.transformer_blocks:
            block.attn.query_chunk_tokens = attention_query_chunk_tokens
            block.attn.key_chunk_tokens = attention_key_chunk_tokens
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size * patch_size, bias=True)
        frequencies = torch.arange(0, 1, 2 * 2 / rotary_dim, dtype=torch.float32)
        self.register_buffer("inv_freq", 1.0 / (rope_theta**frequencies), persistent=False)

    def _rotary_embeddings(
        self,
        height: int,
        width: int,
        batch: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = (torch.arange(height, dtype=torch.float32, device=device) + 0.5) / height * 2.0 - 1.0
        x = (torch.arange(width, dtype=torch.float32, device=device) + 0.5) / width * 2.0 - 1.0
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        position_ids = torch.stack((y_grid, x_grid), dim=-1).reshape(1, height * width, 2)
        suffix_ids = torch.zeros(
            (1, self.num_register_tokens + 1, 2), dtype=torch.float32, device=device
        )
        position_ids = torch.cat((position_ids, suffix_ids), dim=1).expand(batch, -1, -1)
        angles = (
            position_ids[..., None]
            * self.inv_freq.to(device=device)[None, None, None, :]
            * (2.0 * math.pi)
        ).flatten(2, 3)
        angles = angles.repeat(1, 1, 2)
        return angles.cos(), angles.sin()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        interrupt_check: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        if interrupt_check is not None:
            interrupt_check()
        if hidden_states.ndim != 4:
            raise ValueError(
                "MiniMax H3 Image decoder expects [batch, 24, latent_height, latent_width]."
            )
        batch, channels, latent_height, latent_width = hidden_states.shape
        if channels != self.x_embedder.in_features:
            raise ValueError(
                f"MiniMax H3 Image decoder expects {self.x_embedder.in_features} latent channels; got {channels}."
            )

        tokens = hidden_states.permute(0, 2, 3, 1).reshape(
            batch, latent_height * latent_width, channels
        )
        tokens = self.x_embedder(tokens)
        num_patches = tokens.shape[1]
        registers = self.register_tokens.to(dtype=tokens.dtype).expand(batch, -1, -1)
        cls_token = torch.zeros_like(tokens[:, :1])
        tokens = torch.cat((tokens, registers, cls_token), dim=1)
        cos, sin = self._rotary_embeddings(latent_height, latent_width, batch, tokens.device)

        for block in self.transformer_blocks:
            if interrupt_check is not None:
                interrupt_check()
            tokens = block(tokens, cos, sin, interrupt_check=interrupt_check)

        pixels = self.proj_out(self.norm_out(tokens))[:, :num_patches]
        patch = self.patch_size
        pixels = pixels.view(batch, latent_height, latent_width, self.out_channels, patch, patch)
        pixels = pixels.permute(0, 3, 1, 4, 2, 5).contiguous()
        return pixels.reshape(
            batch, self.out_channels, latent_height * patch, latent_width * patch
        )


class MiniMaxH3ImageVAE(nn.Module):
    """H3 Image VAE with RGB `[-1, 1]` inputs and `[0, 1]` RGB outputs.

    ``encode`` accepts `[B,3,H,W]` or `[B,3,1,H,W]` RGB pixels and returns
    fixed-seed sampled, channel-normalized `[B,24,1,H/16,W/16]` latents.
    ``decode`` accepts either the 5D latent shape or its 4D frame form and
    returns `[B,3,H,W]` RGB pixels in `[0,1]`. The H3 image checkpoint uses
    ImageNet pixel normalization internally and H3's published 24-channel
    latent mean/std.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        latent_channels: int = 24,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
        decoder_num_layers: int = 24,
        decoder_num_attention_heads: int = 16,
        decoder_attention_head_dim: int = 64,
        decoder_num_register_tokens: int = 4,
        decoder_ffn_mult: int = 4,
        decoder_rope_theta: float = 100.0,
        decoder_rope_dim_ratio: float = 0.5,
        decoder_norm_eps: float = 1e-5,
        latents_mean: Iterable[float] = IMAGE_VAE_LATENTS_MEAN,
        latents_std: Iterable[float] = IMAGE_VAE_LATENTS_STD,
        encoder_tile_size: int = 512,
        encoder_tile_overlap: int = 128,
        decoder_tile_size: int = 256,
        decoder_tile_overlap: int = 64,
    ) -> None:
        super().__init__()
        if len(block_out_channels) != len(spatial_downsample_factors):
            raise ValueError("Image VAE encoder channel and spatial-downsample levels must match.")
        if len(block_out_channels) != len(temporal_downsample_factors):
            raise ValueError("Image VAE encoder channel and temporal-downsample levels must match.")
        if encoder_tile_size <= 0 or encoder_tile_size % _IMAGE_SPATIAL_COMPRESSION:
            raise ValueError("Image VAE encoder tile size must be a positive multiple of 16 pixels.")
        if encoder_tile_overlap < 0 or encoder_tile_overlap % _IMAGE_SPATIAL_COMPRESSION:
            raise ValueError("Image VAE encoder tile overlap must be a non-negative multiple of 16 pixels.")
        if encoder_tile_overlap >= encoder_tile_size:
            raise ValueError("Image VAE encoder tile overlap must be smaller than its tile size.")
        if decoder_tile_size <= 0 or decoder_tile_size % _IMAGE_SPATIAL_COMPRESSION:
            raise ValueError("Image VAE decoder tile size must be a positive multiple of 16 pixels.")
        if decoder_tile_overlap < 0 or decoder_tile_overlap % _IMAGE_SPATIAL_COMPRESSION:
            raise ValueError("Image VAE decoder tile overlap must be a non-negative multiple of 16 pixels.")
        if decoder_tile_overlap >= decoder_tile_size:
            raise ValueError("Image VAE decoder tile overlap must be smaller than its tile size.")

        self.latent_channels = latent_channels
        self.spatial_compression_ratio = math.prod(spatial_downsample_factors)
        if self.spatial_compression_ratio != _IMAGE_SPATIAL_COMPRESSION:
            raise ValueError(
                f"MiniMax H3 Image requires 16x spatial compression, got {self.spatial_compression_ratio}x."
            )
        self.encoder = MiniMaxH3VideoEncoder3d(
            in_channels=in_channels,
            out_channels=2 * latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            spatial_downsample_factors=spatial_downsample_factors,
            temporal_downsample_factors=temporal_downsample_factors,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = MiniMaxH3ImageViTDecoder(
            in_channels=latent_channels,
            out_channels=3,
            patch_size=self.spatial_compression_ratio,
            num_layers=decoder_num_layers,
            heads=decoder_num_attention_heads,
            head_dim=decoder_attention_head_dim,
            num_register_tokens=decoder_num_register_tokens,
            ffn_mult=decoder_ffn_mult,
            rope_theta=decoder_rope_theta,
            rope_dim_ratio=decoder_rope_dim_ratio,
            norm_eps=decoder_norm_eps,
        )
        self.register_buffer(
            "latents_mean", torch.tensor(tuple(latents_mean), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "latents_std", torch.tensor(tuple(latents_std), dtype=torch.float32), persistent=False
        )
        if self.latents_mean.numel() != latent_channels or self.latents_std.numel() != latent_channels:
            raise ValueError("Image VAE latent mean/std must have one value per latent channel.")
        self.register_buffer(
            "pixel_mean", torch.tensor(_PIXEL_MEAN, dtype=torch.float32).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(_PIXEL_STD, dtype=torch.float32).view(1, 3, 1, 1, 1), persistent=False
        )
        self.encoder_tile_size = encoder_tile_size
        self.encoder_tile_overlap = encoder_tile_overlap
        self.decoder_tile_size = decoder_tile_size
        self.decoder_tile_overlap = decoder_tile_overlap
        self._model_dtype = torch.float32
        self._interrupt = False

    def _check_interrupted(self) -> None:
        if bool(getattr(self, "_interrupt", False)):
            raise InterruptedError("MiniMax H3 Image VAE operation was cancelled.")

    @staticmethod
    def _tile_plan(
        length: int,
        tile_size: int,
        min_overlap: int,
        *,
        alignment: int = _IMAGE_SPATIAL_COMPRESSION,
    ) -> tuple[list[int], list[int]]:
        if alignment <= 0:
            raise ValueError("Image VAE tile alignment must be positive.")
        if length <= tile_size:
            return [0], []

        num_tiles = max(2, math.ceil((length - min_overlap) / (tile_size - min_overlap)))
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) < length:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for index in range(remaining // alignment):
            overlaps[index % (num_tiles - 1)] += alignment

        starts = [0]
        for overlap in overlaps:
            starts.append(starts[-1] + tile_size - overlap)
        return starts, overlaps

    @staticmethod
    def _blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
        extent = min(extent, a.shape[dim], b.shape[dim])
        if extent <= 0:
            return b
        positions = torch.arange(extent, dtype=b.dtype, device=b.device)
        weight_a = (1.0 - positions / extent)
        weight_b = positions / extent
        shape = [1] * b.ndim
        shape[dim] = extent
        weight_a = weight_a.view(shape)
        weight_b = weight_b.view(shape)
        slice_a = [slice(None)] * a.ndim
        slice_b = [slice(None)] * b.ndim
        slice_a[dim] = slice(-extent, None)
        slice_b[dim] = slice(0, extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
        if extent == b.shape[dim]:
            return blended
        rest = [slice(None)] * b.ndim
        rest[dim] = slice(extent, None)
        return torch.cat((blended, b[tuple(rest)]), dim=dim)

    def _encode_moments(self, normalized_pixels: torch.Tensor) -> torch.Tensor:
        height, width = normalized_pixels.shape[-2:]
        tile = self.encoder_tile_size
        if height <= tile and width <= tile:
            self._check_interrupted()
            encoded = self.quant_conv(self.encoder(normalized_pixels))
            self._check_interrupted()
            return encoded

        y_starts, y_overlaps = self._tile_plan(height, tile, self.encoder_tile_overlap)
        x_starts, x_overlaps = self._tile_plan(width, tile, self.encoder_tile_overlap)
        y_lengths = [min(tile, height - start) for start in y_starts]
        x_lengths = [min(tile, width - start) for start in x_starts]
        rows = []
        previous_raw_row: list[torch.Tensor] | None = None
        for y_index, (y_start, y_length) in enumerate(zip(y_starts, y_lengths)):
            untrimmed_row = []
            result_row = []
            for x_index, (x_start, x_length) in enumerate(zip(x_starts, x_lengths)):
                self._check_interrupted()
                tile_pixels = normalized_pixels[
                    ..., y_start : y_start + y_length, x_start : x_start + x_length
                ]
                raw = self.quant_conv(self.encoder(tile_pixels))
                self._check_interrupted()
                stitched = raw
                if previous_raw_row is not None:
                    stitched = self._blend(
                        previous_raw_row[x_index], stitched,
                        y_overlaps[y_index - 1] // self.spatial_compression_ratio, -2,
                    )
                untrimmed_row.append(stitched)
                if x_index:
                    stitched = self._blend(
                        untrimmed_row[x_index - 1], stitched,
                        x_overlaps[x_index - 1] // self.spatial_compression_ratio, -1,
                    )
                if y_index < len(y_starts) - 1:
                    trim_height = y_overlaps[y_index] // self.spatial_compression_ratio
                    if trim_height:
                        stitched = stitched[..., :-trim_height, :]
                if x_index < len(x_starts) - 1:
                    trim_width = x_overlaps[x_index] // self.spatial_compression_ratio
                    if trim_width:
                        stitched = stitched[..., :, :-trim_width]
                result_row.append(stitched)
            rows.append(torch.cat(result_row, dim=-1))
            previous_raw_row = untrimmed_row
        return torch.cat(rows, dim=-2)

    def _decode_pixels(self, unnormalized_latents: torch.Tensor, *, tiled: bool) -> torch.Tensor:
        latent_height, latent_width = unnormalized_latents.shape[-2:]
        tile_latent_size = self.decoder_tile_size // self.spatial_compression_ratio
        overlap_latent = self.decoder_tile_overlap // self.spatial_compression_ratio
        pixel_height = latent_height * self.spatial_compression_ratio
        pixel_width = latent_width * self.spatial_compression_ratio
        if (
            not tiled
            or (pixel_height <= self.decoder_tile_size and pixel_width <= self.decoder_tile_size)
        ):
            self._check_interrupted()
            decoded = self.decoder(
                self.post_quant_conv(unnormalized_latents),
                interrupt_check=self._check_interrupted,
            )
            self._check_interrupted()
            return decoded

        y_starts, y_overlaps = self._tile_plan(
            latent_height, tile_latent_size, overlap_latent, alignment=1
        )
        x_starts, x_overlaps = self._tile_plan(
            latent_width, tile_latent_size, overlap_latent, alignment=1
        )
        y_lengths = [min(tile_latent_size, latent_height - start) for start in y_starts]
        x_lengths = [min(tile_latent_size, latent_width - start) for start in x_starts]
        canvas: torch.Tensor | None = None
        previous_row_strip: torch.Tensor | None = None
        output_y = 0
        for y_index, (y_start, y_length) in enumerate(zip(y_starts, y_lengths)):
            self._check_interrupted()
            next_y_overlap = (
                y_overlaps[y_index] * self.spatial_compression_ratio
                if y_index < len(y_starts) - 1
                else 0
            )
            next_row_strip = None
            if next_y_overlap:
                next_row_strip = torch.empty(
                    unnormalized_latents.shape[0],
                    3,
                    next_y_overlap,
                    pixel_width,
                    dtype=unnormalized_latents.dtype,
                    device=unnormalized_latents.device,
                )
            output_x = 0
            output_row_height = 0
            left_overlap_tail: torch.Tensor | None = None
            for x_index, (x_start, x_length) in enumerate(zip(x_starts, x_lengths)):
                self._check_interrupted()
                latent_tile = unnormalized_latents[
                    ..., y_start : y_start + y_length, x_start : x_start + x_length
                ]
                raw = self.decoder(
                    self.post_quant_conv(latent_tile),
                    interrupt_check=self._check_interrupted,
                )
                self._check_interrupted()
                tile = raw
                tile_width = x_length * self.spatial_compression_ratio
                if y_index and y_overlaps[y_index - 1]:
                    if previous_row_strip is None:
                        raise RuntimeError("Missing previous image VAE decoder overlap strip.")
                    tile = self._blend(
                        previous_row_strip[
                            ..., x_start * self.spatial_compression_ratio :
                            (x_start + x_length) * self.spatial_compression_ratio
                        ],
                        tile,
                        y_overlaps[y_index - 1] * self.spatial_compression_ratio,
                        -2,
                    )
                if x_index and x_overlaps[x_index - 1]:
                    if left_overlap_tail is None:
                        raise RuntimeError("Missing previous image VAE decoder tile overlap.")
                    tile = self._blend(
                        left_overlap_tail,
                        tile,
                        x_overlaps[x_index - 1] * self.spatial_compression_ratio,
                        -1,
                    )

                next_x_overlap = (
                    x_overlaps[x_index] * self.spatial_compression_ratio
                    if x_index < len(x_starts) - 1
                    else 0
                )
                left_overlap_tail = (
                    tile[..., -next_x_overlap:].clone() if next_x_overlap else None
                )
                if next_x_overlap:
                    tile = tile[..., :-next_x_overlap]

                if next_y_overlap:
                    if next_row_strip is None:
                        raise RuntimeError("Could not allocate image VAE decoder overlap strip.")
                    next_row_strip[
                        ..., output_x : output_x + tile.shape[-1]
                    ].copy_(tile[..., -next_y_overlap:, :])
                    tile = tile[..., :-next_y_overlap, :]

                if canvas is None:
                    canvas = torch.empty(
                        *tile.shape[:-2],
                        pixel_height,
                        pixel_width,
                        dtype=tile.dtype,
                        device=tile.device,
                    )
                canvas[
                    ...,
                    output_y : output_y + tile.shape[-2],
                    output_x : output_x + tile.shape[-1],
                ].copy_(tile)
                output_row_height = tile.shape[-2]
                output_x += tile.shape[-1]

            if output_x != pixel_width:
                raise RuntimeError(
                    f"Image VAE decoder tile row covered {output_x} pixels; expected {pixel_width}."
                )
            if output_row_height <= 0:
                raise RuntimeError("Image VAE decoder produced an empty output tile row.")
            previous_row_strip = next_row_strip
            output_y += output_row_height

        if canvas is None or output_y != pixel_height:
            raise RuntimeError(
                f"Image VAE decoder tiles covered {output_y} rows; expected {pixel_height}."
            )
        return canvas

    def _model_run_dtype(self) -> torch.dtype:
        dtype = getattr(self, "_model_dtype", None)
        return dtype if isinstance(dtype, torch.dtype) else torch.float32

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode normalized RGB images to fixed-seed sampled H3 image latents."""

        self._check_interrupted()
        if images.ndim == 4:
            images = images.unsqueeze(2)
        if images.ndim != 5 or images.shape[1] != 3 or images.shape[2] != 1:
            raise ValueError("MiniMax H3 Image encode expects RGB shaped [B,3,H,W] or [B,3,1,H,W].")
        height, width = images.shape[-2:]
        if height % _IMAGE_SPATIAL_COMPRESSION or width % _IMAGE_SPATIAL_COMPRESSION:
            raise ValueError(
                f"MiniMax H3 Image VAE dimensions must be multiples of 16; received {height}x{width}."
            )

        run_dtype = self._model_run_dtype()
        pixels = images.to(dtype=torch.float32).add(1.0).mul_(0.5)
        pixel_mean = self.pixel_mean.to(device=pixels.device, dtype=torch.float32)
        pixel_std = self.pixel_std.to(device=pixels.device, dtype=torch.float32)
        normalized = (pixels - pixel_mean) / pixel_std
        normalized = normalized.to(dtype=run_dtype)
        moments = self._encode_moments(normalized).float()
        mean, logvar = moments.chunk(2, dim=1)

        # Match the published image encoder's fixed seed. A fresh generator per
        # item makes output independent of how references are batched.
        samples = []
        for item_mean, item_logvar in zip(mean, logvar):
            generator = torch.Generator(device="cpu").manual_seed(_IMAGE_ENCODE_SEED)
            noise = torch.randn(
                item_mean.shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(item_mean.device)
            std = torch.exp(0.5 * item_logvar.clamp(-30.0, 20.0))
            samples.append(item_mean + std * noise)
        sampled = torch.stack(samples, dim=0)
        latent_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(sampled)
        latent_std = self.latents_std.view(1, -1, 1, 1, 1).to(sampled)
        return (sampled - latent_mean) / latent_std

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, tiled: bool = True) -> torch.Tensor:
        """Decode normalized H3 image latents to RGB pixels in `[0,1]`.

        The checkpoint's native 256-pixel, 64-pixel-overlap spatial tiling is
        used for large images. Set ``tiled=False`` to run one global ViT pass.
        Images no larger than one tile keep the existing direct decoder path.
        """

        self._check_interrupted()
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)
        if latents.ndim != 5 or latents.shape[1] != self.latent_channels or latents.shape[2] != 1:
            raise ValueError(
                f"MiniMax H3 Image decode expects [B,{self.latent_channels},1,H,W] latents."
            )
        run_dtype = self._model_run_dtype()
        latent_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=run_dtype)
        latent_std = self.latents_std.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=run_dtype)
        unnormalized = latents.to(dtype=run_dtype) * latent_std + latent_mean
        decoded = self._decode_pixels(unnormalized.squeeze(2), tiled=tiled)
        self._check_interrupted()
        pixels = decoded.float() * self.pixel_std.to(device=decoded.device, dtype=torch.float32).squeeze(2)
        return (pixels + self.pixel_mean.to(device=decoded.device, dtype=torch.float32).squeeze(2)).clamp_(0.0, 1.0)


def _coerce_dtype(dtype: torch.dtype | str | None) -> torch.dtype:
    if dtype is None:
        return torch.float16
    if isinstance(dtype, torch.dtype):
        return dtype
    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[str(dtype).lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported MiniMax H3 Image VAE dtype: {dtype!r}.") from error


def preprocess_image_vae_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map only the shared CNN encoder names, preserving native image decoder keys."""

    from .checkpoint import _rename_video_vae_key

    converted = {}
    for source_key, tensor in state_dict.items():
        if source_key in {"decoder.mask_token", "latents_mean", "latents_std"}:
            continue
        if source_key.startswith("encoder.down."):
            source_key = _rename_video_vae_key(source_key)
        converted[source_key] = tensor
    return converted


def load_image_vae(path: str, dtype: torch.dtype | str | None = None) -> MiniMaxH3ImageVAE:
    """Build the VAE on meta and stream the verified FP16 checkpoint via MMGP."""

    from accelerate import init_empty_weights
    from mmgp import offload

    model_dtype = _coerce_dtype(dtype)
    with init_empty_weights(include_buffers=False):
        vae = MiniMaxH3ImageVAE()
    offload.load_model_data(
        vae,
        path,
        writable_tensors=False,
        preprocess_sd=preprocess_image_vae_state_dict,
        default_dtype=model_dtype,
        fp32_dtype=model_dtype,
    )
    vae._model_dtype = model_dtype
    vae._interrupt = False
    return vae.eval().requires_grad_(False)
