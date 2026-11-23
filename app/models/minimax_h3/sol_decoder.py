# Adapted from Sol-H3-VAE (https://huggingface.co/Efficient-Large-Model/Sol-H3-VAE, code/sol_h3_vae), released under the Apache
# License, Version 2.0: the window, global-token and rotary geometry of its noncausal decoder.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP.
# See LICENSE-WanGP-Community-2.0.txt for the WanGP modifications.
"""The windowed Sol-H3 decoder used by the pi-H3 DWD pixel head.

This keeps the upstream ring-buffer and window geometry while using Maestro's
native attention dispatcher and ordinary PyTorch projections. The H3 DWD head
is optional, so the additional decoder is constructed only for its dedicated
model variants.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from shared.attention import pay_attention as _shared_pay_attention
except (ImportError, OSError):
    # The pinned dispatcher depends on MMGP/Quanto extras. The pixel refiner
    # can use ordinary PyTorch SDPA on the legacy backend without them.
    _shared_pay_attention = None

BLOCK, WINDOW, TIME_SHIFT, FRAMES_TO_TRIM = 5, 16, 2, 3
CHUNK_ROWS = 2048
WINDOW_BYTES = 48 << 20
EMIT_BYTES, EMIT_TOKEN_BYTES = 64 << 20, 40 << 10

try:
    from mmgp import transfer as _mmgp_transfer
except (ImportError, AttributeError):
    # Maestro's legacy MMGP tree has no transfer module. DWD still has a
    # bounded band writer in that runtime; only the optional pinned MMGP
    # helper is absent.
    _mmgp_transfer = None


class _TorchRamWriter:
    """Synchronous tensor copy used when MMGP's optional RAM writer is absent."""

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def write(self, destination: torch.Tensor, source: torch.Tensor) -> None:
        source = source.to(
            device=destination.device,
            dtype=destination.dtype,
            non_blocking=False,
        )
        destination.copy_(source)


def _ram_writer(*args, **kwargs):
    writer_type = getattr(_mmgp_transfer, "RamWriter", None)
    if callable(writer_type):
        return writer_type(*args, **kwargs)
    return _TorchRamWriter(*args, **kwargs)


def pay_attention(qkv, **kwargs):
    if _shared_pay_attention is not None:
        return _shared_pay_attention(qkv, **kwargs)
    query, key, value = qkv
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    )
    return output.transpose(1, 2)


class _VideoRotaryPosEmbed(nn.Module):
    """Lightweight three-axis rotary table used by the pixel refiner."""

    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3):
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(
                f"DWD rotary dimension {dim} must be divisible by {2 * num_axes}."
            )
        inv_freq = 1.0 / theta ** torch.arange(
            0, 1, 2 * num_axes / dim, dtype=torch.float32, device="cpu"
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor):
        inv_freq = self.inv_freq.to(device=position_ids.device)
        angles = (
            2.0
            * math.pi
            * position_ids[:, :, :, None]
            * inv_freq[None, None, None, :]
        )
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


def _norm(norm: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    return norm(x.float()).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary_(x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> None:
    cos, sin = (item.to(device=x.device, dtype=x.dtype) for item in rotary)
    rotary_dim = int(cos.shape[-1])
    rotated = x[..., :rotary_dim]
    rotated.copy_(rotated * cos + _rotate_half(rotated) * sin)


class _SolAttention(nn.Module):
    """State-dict compatible fused-QKV attention used by the DWD refiner."""

    def __init__(self, dim: int, heads: int, head_dim: int, eps: float, dtype):
        super().__init__()
        self.heads = heads
        self.dim_head = head_dim
        self.norm_q = nn.RMSNorm(head_dim, eps=eps, elementwise_affine=False, dtype=dtype)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps, elementwise_affine=False, dtype=dtype)
        self.to_qkv = nn.Linear(dim, 3 * heads * head_dim, dtype=dtype)
        self.to_out = nn.Linear(heads * head_dim, dim, dtype=dtype)


class _SolFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dtype):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * dim * mult, dtype=dtype)
        self.w2 = nn.Linear(dim * mult, dim, dtype=dtype)


class _SolBlock(nn.Module):
    """One trained Sol-H3 decoder block with native H3 tensor names."""

    def __init__(self, dim: int, heads: int, head_dim: int, mult: int, eps: float, dtype):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, dtype=dtype)
        self.attn = _SolAttention(dim, heads, head_dim, eps, dtype)
        self.scale1 = nn.Parameter(torch.zeros(dim, dtype=dtype))
        self.norm2 = nn.RMSNorm(dim, eps=eps, dtype=dtype)
        self.ff = _SolFeedForward(dim, mult, dtype)
        self.scale2 = nn.Parameter(torch.zeros(dim, dtype=dtype))

    def _project_qkv(self, x, qkv, rope):
        attn = self.attn
        flat = qkv.view(qkv.shape[0], -1)
        for start in range(0, x.shape[0], CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, x.shape[0])
            normalized = _norm(self.norm1, x[start:stop])
            flat[start:stop] = attn.to_qkv(normalized)
        query, key = qkv[None, :, :, 0], qkv[None, :, :, 1]
        query.copy_(attn.norm_q(query.float()).to(query.dtype))
        key.copy_(attn.norm_k(key.float()).to(key.dtype))
        if rope is not None:
            _apply_rotary_(query, rope)
            _apply_rotary_(key, rope)

    def _post_attention(self, attention, x):
        for start in range(0, x.shape[0], CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, x.shape[0])
            out = x[start:stop]
            out.addcmul_(self.attn.to_out(attention[start:stop]), self.scale1)
            normalized = _norm(self.norm2, out)
            gate, value = self.ff.w1(normalized).chunk(2, dim=-1)
            feed_forward = self.ff.w2(F.silu(gate) * value)
            out.addcmul_(feed_forward, self.scale2)

    def forward(self, state, layer):
        decoder, x, S, HW, starts = state.decoder, state.x, state.S, state.HW, state.starts
        parity = state.parities[layer % 2]
        last = layer == len(decoder.transformer_blocks) - 1
        query, key, value = state.qkv.unbind(2)
        heads, head_dim, dim = decoder.heads, decoder.head_dim, decoder.dim
        base = TIME_SHIFT * HW + S
        for block, lo, hi, windows in parity.steps:
            interrupted = getattr(decoder, "_interrupt", False)
            if (interrupted() if callable(interrupted) else interrupted):
                raise InterruptedError("MiniMax H3 DWD decoding was cancelled.")
            count = 0
            if block is not None:
                count = starts[block + 1] - starts[block]
                if parity.shifted and block > 0:
                    state.qkv[:base].copy_(
                        state.qkv[base + (BLOCK - TIME_SHIFT) * HW:base + BLOCK * HW + S]
                    )
                self._project_qkv(
                    x[starts[block]:starts[block + 1]],
                    state.qkv[base:base + count],
                    parity.rope(block),
                )
                self._project_qkv(
                    x[state.N + block * S:state.N + (block + 1) * S],
                    state.qkv[base + count:base + count + S],
                    None,
                )
            global_out = (
                state.global_attention[block * S:(block + 1) * S]
                if block is not None and not last
                else None
            )
            for queries, keys, outputs in windows:
                n, length = queries.shape
                qkv_list = [
                    query.index_select(0, queries.reshape(-1)).view(n, length, heads, head_dim),
                    key.index_select(0, keys.reshape(-1)).view(n, -1, heads, head_dim),
                    value.index_select(0, keys.reshape(-1)).view(n, -1, heads, head_dim),
                ]
                attended = pay_attention(qkv_list, recycle_q=True).reshape(n * length, dim)
                state.attention.index_copy_(0, outputs.reshape(-1), attended)
            if global_out is not None:
                rows = slice(base, base + count + S)
                qkv_list = [
                    query[base + count:base + count + S][None],
                    key[rows][None],
                    value[rows][None],
                ]
                global_out.copy_(pay_attention(qkv_list, force_attention="sdpa").reshape(S, dim))
            rows = slice(lo * HW, hi * HW)
            self._post_attention(state.attention[:(hi - lo) * HW], x[rows])
            if last:
                decoder._emit(x[rows], lo, hi, state)
            if block is not None and decoder.progress is not None:
                decoder.progress.advance()
        if not last:
            self._post_attention(state.global_attention, x[state.N:])


class _Parity:
    """Precomputed shifted or unshifted local-window layout for one clip."""

    def __init__(self, decoder, state, odd):
        T, H, W, HW, S, nb, starts = state.T, state.H, state.W, state.HW, state.S, state.nb, state.starts
        shift, time_shift = (WINDOW // 2, TIME_SHIFT) if odd else (0, 0)
        axes = torch.meshgrid(
            torch.arange(T, device="cpu"),
            torch.arange(H, device="cpu"),
            torch.arange(W, device="cpu"),
            indexing="ij",
        )
        t, y, x = (axis.reshape(-1) for axis in axes)
        j = (t + time_shift) // BLOCK
        columns, rows = -(-(W + shift) // WINDOW), -(-(H + shift) // WINDOW)
        window = (j * rows + (y + shift) // WINDOW) * columns + (x + shift) // WINDOW
        current = j.clamp(max=nb - 1)
        base = TIME_SHIFT * HW + S
        ring = torch.where(
            t // BLOCK == current,
            base + (t - BLOCK * current) * HW,
            (t - (BLOCK * current - TIME_SHIFT)) * HW,
        ) + y * W + x
        order = torch.argsort(window, stable=True)
        _, lengths = torch.unique_consecutive(window[order], return_counts=True)
        firsts = torch.cumsum(lengths, 0) - lengths
        window_j = j[order[firsts]]
        spans_two = t[order[firsts]] // BLOCK != t[order[firsts + lengths - 1]] // BLOCK
        steps = int(j.max()) + 1
        lows = torch.full((steps,), T, dtype=torch.long, device="cpu").scatter_reduce(0, j, t, "amin")
        highs = torch.zeros(steps, dtype=torch.long, device="cpu").scatter_reduce(0, j, t + 1, "amax")
        self.steps = []
        layouts, first_steps = {}, []
        for step in range(steps):
            block = min(step, nb - 1)
            lo, hi = int(lows[step]), int(highs[step])
            bounds = (lo - block * BLOCK, hi - block * BLOCK, starts[block + 1] - starts[block])
            first = layouts.setdefault(bounds, step)
            first_steps.append(first)
            groups = [] if first == step else self.steps[first][3]
            self.steps.append((block if step < nb else None, lo, hi, groups))
        for step, length, two in sorted(set(zip(window_j.tolist(), lengths.tolist(), spans_two.tolist()))):
            if step != first_steps[step]:
                continue
            chosen = ((window_j == step) & (lengths == length) & (spans_two == two)).nonzero().flatten()
            tokens = order[firsts[chosen][:, None] + torch.arange(length, device="cpu")]
            block = min(step, nb - 1)
            extra = base + starts[block + 1] - starts[block] + torch.arange(S, device="cpu")
            if two:
                extra = torch.cat([TIME_SHIFT * HW + torch.arange(S, device="cpu"), extra])
            keys = torch.cat([ring[tokens], extra.expand(len(chosen), -1)], 1)
            per_call = max(1, WINDOW_BYTES // (keys.shape[1] * decoder.dim * 2))
            for first in range(0, len(chosen), per_call):
                part = slice(first, first + per_call)
                self.steps[step][3].append((ring[tokens[part]], keys[part], tokens[part] - int(lows[step]) * HW))
        self.rope_module, self.H, self.W, self.starts, self.device = decoder.rope, H, W, starts, state.device
        self.shifted, self.shift, self.time_shift, self.tables = odd, shift, time_shift, {}

    def rope(self, block):
        local = tuple(
            (block * BLOCK + index + self.time_shift) % BLOCK
            for index in range((self.starts[block + 1] - self.starts[block]) // (self.H * self.W))
        )
        if local not in self.tables:
            axes = [
                (torch.tensor(local, dtype=torch.float32, device="cpu") + 0.5) / BLOCK * 2 - 1,
                *(((torch.arange(n, dtype=torch.float32, device="cpu") + self.shift + 0.5) / WINDOW * 2 - 1)
                  for n in (self.H, self.W)),
            ]
            ids = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(1, -1, 3).to(self.device)
            self.tables[local] = self.rope_module(ids)
        return self.tables[local]


class _State:
    """Per-clip residual, QKV ring, attention outputs, and window plans."""

    def __init__(self, decoder, z, emit):
        _, _, self.T, self.H, self.W = z.shape
        self.decoder, self.device, self.dtype, self.emit = decoder, z.device, z.dtype, emit
        self.S, self.nb, self.HW = decoder.num_register_tokens + 1, -(-self.T // BLOCK), self.H * self.W
        self.starts = [min(block * BLOCK, self.T) * self.HW for block in range(self.nb + 1)]
        self.N = self.starts[-1]
        self.parities = [_Parity(decoder, self, odd) for odd in (False, True)]
        self.qkv = torch.empty(
            (TIME_SHIFT + BLOCK) * self.HW + 2 * self.S,
            decoder.heads,
            3,
            decoder.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        for parity in self.parities:
            parity.indexed = False
            parity.steps = [
                (block, lo, hi, [tuple(index.to(self.device) for index in group) for group in groups])
                for block, lo, hi, groups in parity.steps
            ]
        self.attention = torch.empty(BLOCK * self.HW, decoder.dim, dtype=self.dtype, device=self.device)
        self.global_attention = torch.empty(self.nb * self.S, decoder.dim, dtype=self.dtype, device=self.device)
        self.x = torch.empty(self.N + self.nb * self.S, decoder.dim, dtype=self.dtype, device=self.device)


class MiniMaxH3SolDecoder3d(nn.Module):
    """Windowed Sol-H3-VAE decoder with bounded output and attention chunks."""

    def __init__(
        self,
        in_channels=24,
        out_channels=3,
        patch_size=16,
        patch_size_t=4,
        num_layers=8,
        num_attention_heads=32,
        attention_head_dim=64,
        num_register_tokens=4,
        ffn_mult=4,
        rope_theta=100.0,
        rope_dim_ratio=0.75,
        norm_eps=1e-5,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.dim = num_attention_heads * attention_head_dim
        self.patch_size, self.patch_size_t, self.out_channels = patch_size, patch_size_t, out_channels
        self.num_register_tokens = num_register_tokens
        self.heads, self.head_dim = num_attention_heads, attention_head_dim
        self.rope = _VideoRotaryPosEmbed(int(attention_head_dim * rope_dim_ratio), theta=rope_theta)
        self.x_embedder = nn.Linear(in_channels, self.dim, dtype=dtype)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, self.dim, dtype=dtype))
        self.transformer_blocks = nn.ModuleList([
            _SolBlock(self.dim, num_attention_heads, attention_head_dim, ffn_mult, norm_eps, dtype)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(self.dim, eps=norm_eps, dtype=torch.float32)
        self.proj_out = nn.Linear(
            self.dim, out_channels * patch_size_t * patch_size * patch_size, dtype=torch.float32
        )
        self.progress = None
        self._interrupt = False

    def output_frames(self, latent_frames):
        return latent_frames * self.patch_size_t - FRAMES_TO_TRIM * -(-latent_frames // BLOCK)

    def _emit(self, x, lo, hi, state):
        patch, patch_t, H, W, HW = self.patch_size, self.patch_size_t, state.H, state.W, state.HW
        band = max(1, EMIT_BYTES // (W * EMIT_TOKEN_BYTES))
        for i in range(hi - lo):
            for top in range(0, H, band):
                bottom = min(top + band, H)
                hidden = F.layer_norm(
                    x[i * HW + top * W:i * HW + bottom * W].float(),
                    (self.dim,), self.norm_out.weight.float(), self.norm_out.bias.float(), self.norm_out.eps,
                )
                pixels = self.proj_out(hidden).view(
                    1, bottom - top, W, self.out_channels, patch_t, patch, patch
                )
                pixels = pixels.permute(0, 3, 4, 1, 5, 2, 6).reshape(
                    1, self.out_channels, patch_t, (bottom - top) * patch, W * patch
                )
                state.emit(
                    pixels[:, :, FRAMES_TO_TRIM:] if (lo + i) % BLOCK == 0 else pixels,
                    top * patch,
                    bottom == H,
                )

    def forward(self, features, grid, emit):
        state = _State(self, features.new_empty((1, 1, *grid)), emit)
        tokens = features.reshape(state.N, -1)
        for start in range(0, state.N, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, state.N)
            state.x[start:stop] = self.x_embedder(tokens[start:stop].to(self.x_embedder.weight.dtype))
        state.x[state.N:] = torch.cat(
            (self.register_tokens[0].to(state.dtype), state.x.new_zeros(1, self.dim))
        ).repeat(state.nb, 1)
        for layer, block in enumerate(self.transformer_blocks):
            block(state, layer)


__all__ = ["MiniMaxH3SolDecoder3d"]
