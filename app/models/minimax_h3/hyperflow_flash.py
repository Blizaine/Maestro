"""Runtime pieces for the distinct HyperFlow H3 Flash checkpoint family.

The schedule and two-time conditioning follow the pinned WanGP snapshot
f4cfc1259215577be9a2ee280c41efbedfb4459a and Video Rebirth's Apache-2.0
HyperFlow H3 Flash schedule. The existing ``hyperflow_recipe`` module remains
the separate eight-step AdaLN-offset LoRA path for ordinary H3 checkpoints.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from mmgp import offload
from .packing import build_row_timesteps

SIGMA_GRIDS = {
    4: (1.0, 0.681872, 0.205019, 0.036496, 0.0),
    6: (1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0),
    8: (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0),
}
HYPERFLOW_STEPS = (4, 6, 8)
HYPERFLOW_DEFAULT_STEPS = 6
HYPERFLOW_VIDEO_SHIFT = 12.0
HYPERFLOW_AUDIO_SHIFT = 3.0


def probe_flash_checkpoint(state_dict, metadata, *, num_layers: int = 50) -> dict:
    """Validate the checkpoint markers and return its model-shaped settings."""
    marker = "endpoint_time_embedder.proj_in.weight"
    if marker not in state_dict:
        return {}
    if metadata is None or "hyperflow_gate" not in metadata:
        raise ValueError("HyperFlow Flash checkpoint metadata is missing its endpoint gate.")
    try:
        endpoint_gate = float(metadata["hyperflow_gate"])
        time_in = state_dict["time_embedder.proj_in.weight"].shape
        endpoint_in = state_dict["endpoint_time_embedder.proj_in.weight"].shape
        time_out = state_dict["time_embedder.proj_out.weight"].shape
        endpoint_out = state_dict["endpoint_time_embedder.proj_out.weight"].shape
        time_embed_dim = int(time_out[0])
    except (KeyError, TypeError, ValueError, OverflowError, IndexError) as error:
        raise ValueError("HyperFlow Flash checkpoint has invalid endpoint-time metadata.") from error
    if (
        not math.isfinite(endpoint_gate)
        or time_embed_dim != 16
        or tuple(time_in) != tuple(endpoint_in)
        or tuple(time_out) != tuple(endpoint_out)
    ):
        raise ValueError("HyperFlow Flash requires compatible endpoint MLPs and 16-wide time embeddings.")
    present_blocks = set()
    low_rank = {}
    for key, tensor in state_dict.items():
        if key.startswith("blocks."):
            parts = key.split(".", 2)
            if len(parts) >= 2 and parts[1].isdecimal():
                index = int(parts[1])
                if not 0 <= index < num_layers:
                    raise ValueError("HyperFlow Flash checkpoint contains an out-of-range block index.")
                present_blocks.add(index)
        if key.endswith(".down.weight"):
            if len(tensor.shape) != 2 or int(tensor.shape[0]) < 1:
                raise ValueError(f"HyperFlow Flash has an invalid low-rank factor shape at {key!r}.")
            low_rank[key[:-len(".down.weight")]] = int(tensor.shape[0])
    if not present_blocks:
        raise ValueError("HyperFlow Flash checkpoint contains no transformer blocks.")
    return {
        "hyperflow_flash": True,
        "endpoint_gate": endpoint_gate,
        "time_embed_dim": time_embed_dim,
        "dropped_blocks": tuple(index for index in range(num_layers) if index not in present_blocks),
        "low_rank": low_rank,
    }


def shift_sigma(sigma: float, shift: float) -> float:
    return float(shift) * float(sigma) / (1.0 + (float(shift) - 1.0) * float(sigma))


def schedule_sigmas(steps: int, shift: float) -> tuple[float, ...]:
    """Shift an upstream Flash grid and retain its final clean endpoint."""
    try:
        grid = SIGMA_GRIDS[int(steps)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"HyperFlow H3 Flash supports {HYPERFLOW_STEPS} steps.") from error
    if isinstance(steps, bool):
        raise ValueError(f"HyperFlow H3 Flash supports {HYPERFLOW_STEPS} steps.")
    return tuple(shift_sigma(sigma, shift) for sigma in grid)


def phase2_tail_sigmas(steps: int, *, shift: float = HYPERFLOW_VIDEO_SHIFT) -> tuple[float, ...]:
    """Return the final N intervals from the eight-step Flash grid."""
    if isinstance(steps, bool):
        raise ValueError("HyperFlow phase 2 supports one to eight evaluations.")
    try:
        count = int(steps)
    except (TypeError, ValueError) as error:
        raise ValueError("HyperFlow phase 2 supports one to eight evaluations.") from error
    if count != steps or not 1 <= count <= 8:
        raise ValueError("HyperFlow phase 2 supports one to eight evaluations.")
    grid = schedule_sigmas(8, shift)
    return tuple(grid[-(count + 1):])


class DroppedBlock(nn.Module):
    """Empty placeholder that preserves original H3 block indices."""


def _materialize_low_rank(value):
    return value.linear.dense_weight() if isinstance(value, LowRankWeight) else value


class LowRankWeight(torch.Tensor):
    """Shape-compatible proxy for a factored checkpoint linear's dense weight."""
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, linear):
        up = linear.up.weight
        return torch.Tensor._make_wrapper_subclass(
            cls,
            (linear.out_features, linear.in_features),
            dtype=up.dtype,
            device=up.device,
        )

    def __init__(self, linear):
        self.linear = linear

    def __repr__(self):
        return f"LowRankWeight(rank {self.linear.rank}, {tuple(self.shape)}, {self.dtype})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return func(
            *tree_map(_materialize_low_rank, args),
            **tree_map(_materialize_low_rank, kwargs or {}),
        )


class LowRankLinear(nn.Linear):
    """Factorized bias-free linear as stored in HyperFlow Flash checkpoints."""
    _mm_requires_native_linear_forward = True

    def __init__(self, in_features, out_features, rank, dtype=None, device=None):
        # Keep the Linear identity used by MMGP's adapter and projection
        # discovery while storing the checkpoint's factors as the parameters.
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.down = nn.Linear(self.in_features, self.rank, bias=False, dtype=dtype, device=device)
        self.up = nn.Linear(self.rank, self.out_features, bias=False, dtype=dtype, device=device)

    @property
    def weight(self):
        return LowRankWeight(self)

    @property
    def bias(self):
        return None

    def forward(self, x):
        return self.up(self.down(x))

    def dense_weight(self):
        eye = torch.eye(self.in_features, dtype=self.up.weight.dtype, device=self.up.weight.device)
        return self.up.weight @ self.down(eye).T

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"


class _LowRankRows:
    @staticmethod
    def supports(module, x):
        return isinstance(module, LowRankLinear) and module.up.weight.dtype == x.dtype

    @staticmethod
    def key(module):
        return module

    @staticmethod
    def prepare(module, x):
        return module.down(x)

    @staticmethod
    def rows(module, prepared, start, stop):
        return F.linear(prepared, module.up.weight[start:stop])

    @staticmethod
    def slice_input(prepared, start, stop):
        return prepared[start:stop]


offload.register_row_projection(_LowRankRows())


def replace_low_rank_linears(model: nn.Module, ranks: dict[str, int], dtype: torch.dtype) -> None:
    """Replace dense modules with checkpoint-shaped ``.down``/``.up`` modules."""
    for name, rank in ranks.items():
        owner_name, _, attr = name.rpartition(".")
        owner = model.get_submodule(owner_name)
        dense = getattr(owner, attr)
        if not isinstance(dense, nn.Linear) or dense.bias is not None:
            raise ValueError(f"Invalid HyperFlow low-rank target {name!r}.")
        setattr(
            owner,
            attr,
            LowRankLinear(dense.in_features, dense.out_features, rank, dtype=dtype, device="meta"),
        )


def _sample_adaln_pairs() -> torch.Tensor:
    """Sample every (t, r) pair used by Flash phase 1 and conditioning rows."""
    pairs = set()
    for grid in SIGMA_GRIDS.values():
        for shift in (HYPERFLOW_VIDEO_SHIFT, HYPERFLOW_AUDIO_SHIFT):
            sigmas = tuple(shift_sigma(sigma, shift) for sigma in grid)
            for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
                pairs.add((1.0 - sigma, 1.0 - next_sigma))
                if shift == HYPERFLOW_VIDEO_SHIFT:
                    from .packing import MINIMAX_H3_KEYFRAME_NOISE_AUG

                    condition = max(1.0 - sigma, MINIMAX_H3_KEYFRAME_NOISE_AUG)
                    pairs.add((condition, condition))
    pairs.add((1.0, 1.0))
    return torch.tensor(sorted(pairs), dtype=torch.float32, device="cpu")


def _cpu_time_embedding(embedder, timesteps: torch.Tensor) -> torch.Tensor:
    half = int(embedder.input_dim) // 2
    times = timesteps.float().cpu().unsqueeze(1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32)
        / half
    )
    embedding = torch.cat(((times * frequencies).cos(), (times * frequencies).sin()), dim=-1)
    weight = lambda layer: (layer.weight.detach().float().cpu(), layer.bias.detach().float().cpu())
    return F.linear(F.silu(F.linear(embedding, *weight(embedder.proj_in))), *weight(embedder.proj_out))


def _adaln_input_map(model: nn.Module, architecture: str) -> torch.Tensor:
    """Fit the base rank-eight AdaLN input onto HyperFlow's two-time MLP."""
    cached = model.__dict__.get("_hyperflow_flash_adaln_map")
    if cached is not None and cached[0] == architecture:
        return cached[1]
    from .lora_affine import _load_affine_package

    table = _load_affine_package(architecture, 8)[0].double()
    pairs = _sample_adaln_pairs()
    t, r = pairs[:, 0], pairs[:, 1]
    current = _cpu_time_embedding(model.time_embedder, t)
    endpoint = _cpu_time_embedding(model.endpoint_time_embedder, r)
    hyperflow_input = F.silu(current + (endpoint - current) * float(model.endpoint_gate)).double()
    position = t.double().clamp(0.0, 1.0) * (table.shape[0] - 1)
    lower = position.floor().long().clamp(max=table.shape[0] - 2)
    curve = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
    design = torch.cat((hyperflow_input, hyperflow_input.new_ones(hyperflow_input.shape[0], 1)), dim=1)
    mapping = torch.linalg.lstsq(design, curve, rcond=1e-12).solution.float()
    error = (
        torch.linalg.vector_norm(design @ mapping.double() - curve)
        / torch.linalg.vector_norm(curve - curve.mean(0))
    ).item()
    print(f"[MiniMax H3 HyperFlow Flash] Fitted LoRA AdaLN map over {len(pairs)} sampled (t, r) pairs; relative error {error:.3g}.")
    model.__dict__["_hyperflow_flash_adaln_map"] = (architecture, mapping)
    return mapping


def convert_adaln_loras(model: nn.Module, model_type: str, state_dict: dict) -> int:
    """Map ordinary H3 AdaLN LoRAs onto Flash's two-time 16-wide input."""
    from .lora_affine import (
        _LORA_SUFFIXES,
        _add_bias_delta,
        _architecture,
        _load_affine_package,
        convert_adaln_loras as convert_to_pruned_curve,
    )

    architecture = _architecture(model_type)
    curve = _load_affine_package(architecture, 8)[0]
    count, _, _, _ = convert_to_pruned_curve(model_type, state_dict, curve)
    mapping = None
    converted = 0
    for down_suffix, up_suffix in _LORA_SUFFIXES:
        marker = "." + down_suffix
        for down_key in [key for key in state_dict if key.endswith(marker) and ".adaln_proj.linear." in key]:
            module_name = down_key[:-len(marker)]
            up_key = module_name + "." + up_suffix
            if up_key not in state_dict:
                raise ValueError(f"MiniMax H3 HyperFlow Flash LoRA is missing {up_key}.")
            if mapping is None:
                mapping = _adaln_input_map(model, architecture)
            mapped = state_dict[down_key].float() @ mapping.T
            state_dict[down_key] = mapped[:, :-1].contiguous()
            _add_bias_delta(state_dict, module_name + ".diff_b", state_dict[up_key].float() @ mapped[:, -1])
            converted += 1
    for diff_key in [key for key in state_dict if key.endswith(".adaln_proj.linear.diff")]:
        if mapping is None:
            mapping = _adaln_input_map(model, architecture)
        mapped = state_dict[diff_key].float() @ mapping.T
        state_dict[diff_key] = mapped[:, :-1].contiguous()
        _add_bias_delta(state_dict, diff_key[:-len(".diff")] + ".diff_b", mapped[:, -1])
        converted += 1
    return converted if converted else count


def two_time_rows(layout, timestep, timestep_indices, endpoints, condition_video_timestep, pinned=()):
    """Produce unique (t, r) conditioning rows matching the upstream rule.

    ``r`` is each generated media row's next time; text follows video, while
    references and target rows held at their own time keep r=t. ``pinned`` is
    reserved for the upstream source-edit path and contains (start, stop, t)
    spans in packed-row order.
    """
    rows_t = timestep[timestep_indices]
    video_r, audio_r = endpoints
    end, end_indices = build_row_timesteps(
        layout,
        float(video_r),
        float(audio_r),
        float(condition_video_timestep),
        1.0,
    )
    rows_r = end.to(rows_t.device)[end_indices.to(rows_t.device)]
    if pinned:
        rows_t = rows_t.clone()
        rows_r = rows_r.clone()
        for start, stop, value in pinned:
            rows_t[start:stop] = float(value)
            rows_r[start:stop] = float(value)
    unique, indices = torch.unique(
        torch.stack((rows_t, rows_r), dim=1), dim=0, sorted=True, return_inverse=True,
    )
    return unique[:, 0].contiguous(), unique[:, 1].contiguous(), indices


__all__ = [
    "DroppedBlock", "HYPERFLOW_AUDIO_SHIFT", "HYPERFLOW_DEFAULT_STEPS", "HYPERFLOW_STEPS",
    "HYPERFLOW_VIDEO_SHIFT", "LowRankLinear", "LowRankWeight", "SIGMA_GRIDS",
    "convert_adaln_loras", "phase2_tail_sigmas", "probe_flash_checkpoint", "replace_low_rank_linears", "schedule_sigmas", "two_time_rows",
]
