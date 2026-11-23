# Adapted from LakonLab (https://github.com/Lakonik/LakonLab), released under the Apache License, Version 2.0: pi-H3's schedule and
# its Decoding Without Decoder pixel head (lakonlab/models/architectures/dwd_head.py, sol_h3/sol_h3.py).
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
"""The trained pi-H3 DWD head and few-step schedules for MiniMax H3 FL2VA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sol_decoder import (
    EMIT_BYTES,
    EMIT_TOKEN_BYTES,
    MiniMaxH3SolDecoder3d,
    _State,
    _ram_writer,
)


PI_MAX_RAW_T = {2: 0.93, 4: 0.96}
PI_FINAL_STEP_SCALE = 0.6


def pi_h3_sigmas(steps, shift=12.0):
    """Video sigma values from the official pi-H3 2- or 4-step training grid."""
    steps = int(steps)
    if steps not in PI_MAX_RAW_T:
        raise ValueError("pi-H3 supports only its trained 2-step and 4-step schedules.")
    shift = float(shift)
    if shift <= 0:
        raise ValueError("pi-H3 flow shift must be positive.")
    first = PI_MAX_RAW_T[steps]
    raw = torch.linspace(
        first,
        first * PI_FINAL_STEP_SCALE / (steps - 1 + PI_FINAL_STEP_SCALE),
        steps,
        dtype=torch.float64,
        device="cpu",
    )
    return (shift * raw / (1.0 + (shift - 1.0) * raw)).tolist()


class _AdaLN(nn.Module):
    def __init__(self, time_dim, hidden):
        super().__init__()
        self.linear = nn.Linear(time_dim, 2 * hidden, dtype=torch.float32)
        self.linear._lock_dtype = torch.float32


class MiniMaxH3DWDRefiner(MiniMaxH3SolDecoder3d):
    """The trained final Sol-H3 decoder layers applied to H3 DiT features."""

    def __init__(self, in_channels=5376, expansion=2, num_layers=8, dtype=torch.bfloat16):
        super().__init__(
            num_layers=num_layers,
            dtype=dtype,
        )
        del self.x_embedder
        self.expansion = expansion
        self.feature_expand = nn.Linear(
            in_channels,
            self.dim * expansion * expansion,
            dtype=dtype,
        )
        self.first_frame_proj_out = nn.Linear(
            self.dim,
            self.out_channels * self.patch_size * self.patch_size,
            dtype=torch.float32,
        )
        self.first_frame_proj_out._lock_dtype = torch.float32

    def reset_rope(self):
        """Recompute the nonpersistent rotary frequencies for an empty-weight load."""
        self.rope = type(self.rope)(int(self.head_dim * 0.75), theta=100.0)

    def _emit(self, x, lo, hi, state):
        patch, H, W, HW = self.patch_size, state.H, state.W, state.HW
        band = max(1, EMIT_BYTES // max(1, W * EMIT_TOKEN_BYTES))
        for i in range(hi - lo):
            first_frame = (lo + i) % 5 == 0
            head = self.first_frame_proj_out if first_frame else self.proj_out
            frame_count = 1 if first_frame else self.patch_size_t
            for top in range(0, H, band):
                bottom = min(top + band, H)
                hidden = F.layer_norm(
                    x[i * HW + top * W:i * HW + bottom * W].float(),
                    (self.dim,),
                    self.norm_out.weight.float(),
                    self.norm_out.bias.float(),
                    self.norm_out.eps,
                )
                pixels = head(hidden.to(head.weight.dtype)).view(
                    1,
                    bottom - top,
                    W,
                    self.out_channels,
                    frame_count,
                    patch,
                    patch,
                )
                pixels = pixels.permute(0, 3, 4, 1, 5, 2, 6).reshape(
                    1,
                    self.out_channels,
                    frame_count,
                    (bottom - top) * patch,
                    W * patch,
                )
                state.emit(pixels, top * patch, bottom == H)

    def forward(self, features_list, grid, emit):
        features = features_list.pop()
        latent_t, latent_h, latent_w = grid
        expansion = self.expansion
        state = _State(
            self,
            features.new_empty(
                (1, 1, latent_t, latent_h * expansion, latent_w * expansion)
            ),
            emit,
        )
        tokens, dtype = latent_h * latent_w, self.feature_expand.weight.dtype
        for frame in range(latent_t):
            expanded = self.feature_expand(
                features[frame * tokens:(frame + 1) * tokens].to(dtype)
            ).view(latent_h, latent_w, expansion, expansion, self.dim)
            state.x[
                frame * state.HW:(frame + 1) * state.HW
            ].view(latent_h, expansion, latent_w, expansion, self.dim).copy_(
                expanded.permute(0, 2, 1, 3, 4)
            )
            del expanded
        del features
        state.x[state.N:] = torch.cat(
            (self.register_tokens[0].to(state.dtype), state.x.new_zeros(1, self.dim))
        ).repeat(state.nb, 1)
        for layer, block in enumerate(self.transformer_blocks):
            block(state, layer)


class MiniMaxH3DWDHead(nn.Module):
    def __init__(self, hidden=5376, time_dim=8, table_size=1025, eps=1e-5, dtype=torch.bfloat16):
        super().__init__()
        self.register_buffer("time_table", torch.empty(table_size, time_dim, dtype=torch.float32))
        self.adaln_proj = _AdaLN(time_dim, hidden)
        self.eps = eps
        self.refiner = MiniMaxH3DWDRefiner(hidden, dtype=dtype)
        self.refiner.norm_out._lock_dtype = torch.float32
        self.refiner.proj_out._lock_dtype = torch.float32

    def features(self, hidden, timestep, row_indices, dtype):
        """Apply the DWD time-table AdaLN to rows selected from the DiT stream.

        `row_indices` is one timestep index per selected video row. This keeps
        timed frame conditioning intact without making a full-size FP32 copy.
        """
        if torch.is_tensor(row_indices):
            row_indices = row_indices.to(device=timestep.device, dtype=torch.long)
            position = timestep.float().flatten().index_select(0, row_indices).clamp_(0.0, 1.0)
            position = position * (self.time_table.shape[0] - 1)
            lower = position.floor().long().clamp_(max=self.time_table.shape[0] - 2)
            fraction = (position - lower).unsqueeze(-1)
            temb = torch.lerp(
                self.time_table.index_select(0, lower),
                self.time_table.index_select(0, lower + 1),
                fraction,
            )
            scale, shift = self.adaln_proj.linear(F.silu(temb)).chunk(2, dim=-1)
            output = torch.empty(hidden.shape, dtype=dtype, device=hidden.device)
            for start in range(0, hidden.shape[0], 8192):
                stop = min(start + 8192, hidden.shape[0])
                output[start:stop] = F.layer_norm(
                    hidden[start:stop].float(), (hidden.shape[-1],), None, None, self.eps
                ).mul_(1.0 + scale[start:stop]).add_(shift[start:stop])
            return output

        # Keep the upstream scalar / span interface available for downstream
        # WanGP callers and simple CPU tests.
        position = timestep.float().flatten().clamp(0.0, 1.0) * (self.time_table.shape[0] - 1)
        lower = position.floor().long().clamp(max=self.time_table.shape[0] - 2)
        temb = torch.lerp(
            self.time_table[lower],
            self.time_table[lower + 1],
            (position - lower).unsqueeze(1),
        )
        scale, shift = self.adaln_proj.linear(F.silu(temb)).chunk(2, dim=-1)
        output = torch.empty(hidden.shape, dtype=dtype, device=hidden.device)
        spans = row_indices if isinstance(row_indices, list) else [(0, hidden.shape[0], row_indices)]
        for first, last, index in spans:
            output[first:last] = F.layer_norm(
                hidden[first:last].float(), (hidden.shape[-1],), None, None, self.eps
            ).mul_(1.0 + scale[index]).add_(shift[index])
        return output

    def decode(self, features_list, grid, convert, interrupt=False):
        refiner = self.refiner
        refiner._interrupt = interrupt
        frames = refiner.output_frames(grid[0])
        height = grid[1] * refiner.expansion * refiner.patch_size
        decoded, position = None, 0
        with _ram_writer("decoded video", depth=64, max_bytes=256 << 20) as writer:
            def emit(pixels, row, last):
                nonlocal decoded, position
                part = convert(pixels)
                if decoded is None:
                    decoded = torch.empty(
                        1,
                        part.shape[1],
                        frames,
                        height,
                        part.shape[-1],
                        dtype=part.dtype,
                        device="cpu",
                    )
                writer.write(
                    decoded[:, :, position:position + part.shape[2], row:row + part.shape[-2]],
                    part,
                )
                if last:
                    position += part.shape[2]

            refiner(features_list, grid, emit)
        if position != frames:
            raise RuntimeError(f"MiniMax H3 DWD head decoded {position} frames, expected {frames}")
        return decoded


__all__ = ["MiniMaxH3DWDHead", "pi_h3_sigmas"]
