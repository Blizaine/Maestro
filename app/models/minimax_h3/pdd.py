# Copyright 2026 Alibaba PAI and VideoX-Fun contributors.
# Copyright 2026 Maestro contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Parallel Decoding Distillation support for MiniMax H3 Acc-LoRAs.

Alibaba PAI's H3 acceleration adapters are not ordinary PEFT LoRAs.  Their
backbone tensors are low-rank updates, but the video and audio output layers
contain one trained head for every interval of a 32-interval schedule.  Four
adjacent interval heads are fused for each model evaluation, producing the
published eight-step recipe.

The schedule and head-fusion math follow Alibaba PAI's Apache-2.0 reference
implementation:
https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/blob/main/minimax_h3_pdd.py

Maestro keeps the large backbone updates in MMGP's normal LoRA/offload path.
Only the two small output-head banks are handled here, pre-fused on CPU into
eight heads and copied one at a time during denoising.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


PDD_NUM_INTERVALS = 32
PDD_BLOCK_SIZE = 4
PDD_NUM_EVALUATIONS = PDD_NUM_INTERVALS // PDD_BLOCK_SIZE
PDD_VIDEO_SHIFT = 12.0
PDD_AUDIO_SHIFT = 3.0

_PDD_HEAD_KEYS = (
    "proj_out.weight",
    "proj_out.bias",
    "audio_proj_out.weight",
    "audio_proj_out.bias",
)


def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    return shift * sigma / (1 + (shift - 1) * sigma)


def pdd_time_grid(shift: float, num_steps: int = PDD_NUM_INTERVALS) -> torch.Tensor:
    """Return Alibaba's ascending MiniMax H3 PDD time grid."""

    sigma = torch.linspace(1.0, 0.0, int(num_steps) + 1, dtype=torch.float64)
    return 1.0 - shifted_sigma(float(shift), sigma)


def pdd_sampling_plan(
    step_sizes: torch.Tensor,
    start: int,
    block_size: int = PDD_BLOCK_SIZE,
) -> torch.Tensor:
    """Return normalized interval weights for one Euler model evaluation."""

    start = int(start)
    block_size = int(block_size)
    plan = torch.zeros(step_sizes.shape[0], dtype=step_sizes.dtype)
    block = step_sizes[start : start + block_size]
    span = block.sum()
    if block.numel() != block_size or float(span) <= 0:
        raise ValueError(
            f"Invalid PDD block start={start}, size={block_size}, "
            f"intervals={step_sizes.shape[0]}."
        )
    plan[start : start + block_size] = block / span
    return plan


def pdd_sampling_plans_for_sigmas(
    sigmas: torch.Tensor | list[float],
    shift: float,
    num_steps: int = PDD_NUM_INTERVALS,
) -> torch.Tensor:
    """Map actual scheduler ranges onto the trained PDD interval heads.

    PDD was trained on ``num_steps`` fine rectified-flow intervals.  A runtime
    evaluation may span any contiguous portion of that grid, so its output
    head must be the overlap-weighted blend of every trained interval it
    covers.  Building plans from the scheduler's real sigma boundaries keeps
    Maestro aligned with WanGP 12.645 and also remains correct if H3's runtime
    schedule is customized in the future.
    """

    boundaries = torch.as_tensor(sigmas).flatten().detach().to(
        device="cpu",
        dtype=torch.float64,
        non_blocking=False,
    )
    if boundaries.numel() < 2:
        raise ValueError("MiniMax H3 PDD requires at least two sigma boundaries.")
    times = 1.0 - boundaries
    fine = pdd_time_grid(float(shift), int(num_steps))
    fine_starts = fine[:-1]
    fine_ends = fine[1:]
    plans: list[torch.Tensor] = []
    for start, end in zip(times[:-1], times[1:]):
        span = end - start
        if float(span) <= 0:
            raise ValueError(
                "MiniMax H3 PDD requires strictly descending sigma boundaries."
            )
        overlap = (
            torch.minimum(fine_ends, end)
            - torch.maximum(fine_starts, start)
        ).clamp_min_(0.0)
        if not torch.isclose(
            overlap.sum(),
            span,
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError(
                "MiniMax H3 PDD sigma interval "
                f"[{float(1.0 - start):.6f}, {float(1.0 - end):.6f}] "
                "is outside its trained grid."
            )
        plans.append(overlap / span)
    return torch.stack(plans)


def is_pdd_state_dict(state_dict: dict) -> bool:
    """Recognize the official PDD tensor layout without relying on a filename."""

    video = state_dict.get("proj_out.weight")
    audio = state_dict.get("audio_proj_out.weight")
    return (
        torch.is_tensor(video)
        and torch.is_tensor(audio)
        and video.ndim == 3
        and audio.ndim == 3
        and int(video.shape[0]) == PDD_NUM_INTERVALS
        and int(audio.shape[0]) == PDD_NUM_INTERVALS
    )


def _local_pdd_module_name(official_name: str) -> str:
    name = str(official_name)
    if name.startswith("transformer_blocks."):
        name = "blocks." + name[len("transformer_blocks.") :]
    elif name.startswith("token_refiner.refiner_blocks."):
        name = "token_refiner.blocks." + name[
            len("token_refiner.refiner_blocks.") :
        ]
    name = name.replace(".attn.to_out.0", ".attn.out_proj")
    name = name.replace(".ff.net.0.proj", ".mlp.fc1")
    name = name.replace(".ff.net.2", ".mlp.fc2")
    name = name.replace(".attn.to_q", ".attn.q_proj")
    name = name.replace(".attn.to_k", ".attn.k_proj")
    name = name.replace(".attn.to_v", ".attn.v_proj")
    return name


def _combine_fused_qkv_lora(
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Represent three independent rank-R adapters as one rank-3R adapter."""

    ordered = [factors[name] for name in ("q", "k", "v")]
    downs = [item[0] for item in ordered]
    ups = [item[1] for item in ordered]
    input_width = int(downs[0].shape[1])
    rank = int(downs[0].shape[0])
    output_width = int(ups[0].shape[0])
    if any(
        tuple(down.shape) != (rank, input_width)
        or tuple(up.shape) != (output_width, rank)
        for down, up in ordered
    ):
        raise ValueError("PDD Q/K/V LoRA factors do not share one compatible shape.")

    combined_down = torch.cat(downs, dim=0).contiguous()
    combined_up = ups[0].new_zeros(
        (output_width * 3, rank * 3),
    )
    for index, up in enumerate(ups):
        combined_up[
            index * output_width : (index + 1) * output_width,
            index * rank : (index + 1) * rank,
        ].copy_(up)
    return combined_down, combined_up


def preprocess_pdd_lora_state_dict(
    state_dict: dict,
    *,
    split_qkv: bool,
) -> dict:
    """Map official Diffusers PDD names to Maestro/MMGP LoRA names.

    Output-head banks are removed here and loaded separately by
    :func:`install_pdd_parallel_heads`.  Split INT8 ConvRot transformers keep
    the three independent Q/K/V adapters.  Legacy fused checkpoints receive
    an exactly equivalent rank-3R block-diagonal adapter.
    """

    if not is_pdd_state_dict(state_dict):
        return state_dict

    converted: dict[str, torch.Tensor] = {}
    fused_groups: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for key, tensor in state_dict.items():
        if key in _PDD_HEAD_KEYS:
            continue
        if key.endswith(".lora_down"):
            official_module = key[: -len(".lora_down")]
            factor = "down"
        elif key.endswith(".lora_up"):
            official_module = key[: -len(".lora_up")]
            factor = "up"
        else:
            # The official checkpoint contains only the four PDD heads and
            # low-rank factors.  Preserve an actionable failure if that
            # contract changes upstream instead of silently dropping tensors.
            raise ValueError(f"Unexpected MiniMax H3 PDD tensor: {key}")

        local_module = _local_pdd_module_name(official_module)
        qkv_kind = None
        for kind in ("q", "k", "v"):
            suffix = f".attn.{kind}_proj"
            if local_module.endswith(suffix):
                qkv_kind = kind
                qkv_parent = local_module[: -len(suffix)]
                break
        if qkv_kind is not None and not split_qkv:
            group = fused_groups.setdefault(qkv_parent, {})
            group.setdefault(qkv_kind, {})[factor] = tensor
            continue

        suffix = "lora_A.weight" if factor == "down" else "lora_B.weight"
        converted[f"{local_module}.{suffix}"] = tensor

    for parent, group in fused_groups.items():
        missing = [
            f"{kind}.{factor}"
            for kind in ("q", "k", "v")
            for factor in ("down", "up")
            if factor not in group.get(kind, {})
        ]
        if missing:
            raise ValueError(
                f"PDD fused QKV adapter '{parent}' is incomplete: "
                + ", ".join(missing)
            )
        down, up = _combine_fused_qkv_lora(
            {
                kind: (group[kind]["down"], group[kind]["up"])
                for kind in ("q", "k", "v")
            }
        )
        converted[f"{parent}.attn.qkv_proj.lora_A.weight"] = down
        converted[f"{parent}.attn.qkv_proj.lora_B.weight"] = up

    return converted


def _fuse_interval_heads_for_plans(
    weights: torch.Tensor,
    biases: torch.Tensor | None,
    plans: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None]:
    num_steps = int(weights.shape[0])
    plans = torch.as_tensor(plans, dtype=torch.float64, device="cpu")
    if plans.ndim != 2 or int(plans.shape[1]) != num_steps:
        raise ValueError(
            "PDD plans must have shape "
            f"(evaluations, {num_steps}), got {tuple(plans.shape)}."
        )
    fused_weights: list[torch.Tensor] = []
    fused_biases: list[torch.Tensor] = []
    for plan in plans:
        active_indices = torch.nonzero(plan > 0, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            raise ValueError("A MiniMax H3 PDD plan cannot be empty.")
        weight = torch.zeros_like(weights[0], dtype=torch.float32)
        for interval in active_indices.tolist():
            weight.add_(
                weights[interval].to(torch.float32),
                alpha=float(plan[interval]),
            )
        fused_weights.append(weight.to(weights.dtype).contiguous())
        if biases is not None:
            bias = torch.zeros_like(biases[0], dtype=torch.float32)
            for interval in active_indices.tolist():
                bias.add_(
                    biases[interval].to(torch.float32),
                    alpha=float(plan[interval]),
                )
            fused_biases.append(bias.to(biases.dtype).contiguous())
    return (
        tuple(fused_weights),
        tuple(fused_biases) if biases is not None else None,
    )


def _fixed_pdd_plans(
    shift: float,
    num_steps: int,
    block_size: int,
) -> torch.Tensor:
    if num_steps % int(block_size):
        raise ValueError(
            f"PDD interval count {num_steps} is not divisible by block size {block_size}."
        )
    step_sizes = pdd_time_grid(float(shift), num_steps).diff()
    return torch.stack(
        [
            pdd_sampling_plan(step_sizes, start, int(block_size))
            for start in range(0, num_steps, int(block_size))
        ]
    )


class MiniMaxH3PDDParallelHead(nn.Module):
    """One MMGP-managed base head plus eight CPU-resident fused PDD heads."""

    def __init__(
        self,
        base: nn.Module,
        weights: tuple[torch.Tensor, ...],
        biases: tuple[torch.Tensor, ...] | None,
        strength: float,
    ):
        super().__init__()
        self.base = base
        self._pdd_weights = weights
        self._pdd_biases = biases
        self.strength = float(strength)
        self.step_index = 0

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    @property
    def num_steps(self) -> int:
        return len(self._pdd_weights)

    def set_step(self, index: int) -> None:
        index = int(index)
        if not 0 <= index < self.num_steps:
            raise IndexError(
                f"PDD head step {index} is outside 0-{self.num_steps - 1}."
            )
        self.step_index = index

    def set_fused_heads(
        self,
        weights: tuple[torch.Tensor, ...],
        biases: tuple[torch.Tensor, ...] | None,
    ) -> None:
        if not weights:
            raise ValueError("MiniMax H3 PDD requires at least one fused head.")
        if biases is not None and len(biases) != len(weights):
            raise ValueError("MiniMax H3 PDD weight and bias plans do not align.")
        self._pdd_weights = weights
        self._pdd_biases = biases
        self.step_index = min(self.step_index, len(weights) - 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self._pdd_weights[self.step_index].to(
            device=hidden_states.device,
            non_blocking=True,
        )
        bias = None
        if self._pdd_biases is not None:
            bias = self._pdd_biases[self.step_index].to(
                device=hidden_states.device,
                non_blocking=True,
            )
        pdd_output = F.linear(hidden_states.to(weight.dtype), weight, bias)
        if self.strength == 1.0:
            return pdd_output
        base_output = self.base(hidden_states)
        return torch.lerp(
            base_output,
            pdd_output.to(base_output.dtype),
            self.strength,
        )


@dataclass
class MiniMaxH3PDDController:
    video_head: MiniMaxH3PDDParallelHead
    audio_head: MiniMaxH3PDDParallelHead
    checkpoint: str
    video_interval_weights: torch.Tensor
    video_interval_biases: torch.Tensor | None
    audio_interval_weights: torch.Tensor
    audio_interval_biases: torch.Tensor | None

    @property
    def num_steps(self) -> int:
        return self.video_head.num_steps

    def set_step(self, index: int) -> None:
        self.video_head.set_step(index)
        self.audio_head.set_step(index)

    def configure_sigmas(
        self,
        video_sigmas: torch.Tensor | list[float],
        audio_sigmas: torch.Tensor | list[float],
    ) -> None:
        """Fuse interval heads for the exact runtime video/audio schedules."""

        video_plans = pdd_sampling_plans_for_sigmas(
            video_sigmas,
            PDD_VIDEO_SHIFT,
            int(self.video_interval_weights.shape[0]),
        )
        audio_plans = pdd_sampling_plans_for_sigmas(
            audio_sigmas,
            PDD_AUDIO_SHIFT,
            int(self.audio_interval_weights.shape[0]),
        )
        if int(video_plans.shape[0]) != int(audio_plans.shape[0]):
            raise ValueError(
                "MiniMax H3 PDD video and audio schedules have different lengths."
            )
        video_fused = _fuse_interval_heads_for_plans(
            self.video_interval_weights,
            self.video_interval_biases,
            video_plans,
        )
        audio_fused = _fuse_interval_heads_for_plans(
            self.audio_interval_weights,
            self.audio_interval_biases,
            audio_plans,
        )
        self.video_head.set_fused_heads(*video_fused)
        self.audio_head.set_fused_heads(*audio_fused)


def install_pdd_parallel_heads(
    transformer: nn.Module,
    checkpoint: str,
    *,
    strength: float = 1.0,
    video_shift: float = PDD_VIDEO_SHIFT,
    audio_shift: float = PDD_AUDIO_SHIFT,
) -> MiniMaxH3PDDController:
    """Load and install the official PDD output heads for one H3 job."""

    from safetensors import safe_open

    release_pdd_parallel_heads(transformer)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        num_steps = int(metadata.get("pdd_num_steps") or PDD_NUM_INTERVALS)
        block_size = int(metadata.get("pdd_block_size") or PDD_BLOCK_SIZE)
        if num_steps != PDD_NUM_INTERVALS or block_size != PDD_BLOCK_SIZE:
            raise ValueError(
                "Maestro currently supports the published MiniMax H3 PDD "
                f"32x4 schedule, got {num_steps}x{block_size}."
            )
        # Detach the interval heads from safetensors' file mapping.  The
        # controller keeps these small banks for scheduler-aware re-fusion;
        # owning the storage lets Windows release/update the checkpoint as
        # soon as this context exits.
        video_weights = handle.get_tensor("proj_out.weight").clone()
        video_biases = handle.get_tensor("proj_out.bias").clone()
        audio_weights = handle.get_tensor("audio_proj_out.weight").clone()
        audio_biases = handle.get_tensor("audio_proj_out.bias").clone()

    video_fused = _fuse_interval_heads_for_plans(
        video_weights,
        video_biases,
        _fixed_pdd_plans(
            float(video_shift),
            int(video_weights.shape[0]),
            block_size,
        ),
    )
    audio_fused = _fuse_interval_heads_for_plans(
        audio_weights,
        audio_biases,
        _fixed_pdd_plans(
            float(audio_shift),
            int(audio_weights.shape[0]),
            block_size,
        ),
    )
    video_head = MiniMaxH3PDDParallelHead(
        transformer.final_layer.video_out,
        video_fused[0],
        video_fused[1],
        strength,
    )
    audio_head = MiniMaxH3PDDParallelHead(
        transformer.final_layer.audio_out,
        audio_fused[0],
        audio_fused[1],
        strength,
    )
    transformer.final_layer.video_out = video_head
    transformer.final_layer.audio_out = audio_head
    controller = MiniMaxH3PDDController(
        video_head=video_head,
        audio_head=audio_head,
        checkpoint=os.path.abspath(str(checkpoint)),
        video_interval_weights=video_weights,
        video_interval_biases=video_biases,
        audio_interval_weights=audio_weights,
        audio_interval_biases=audio_biases,
    )
    transformer._pdd_controller = controller
    print(
        "[MiniMax H3 PDD] Loaded Alibaba PAI acceleration adapter: "
        f"{os.path.basename(str(checkpoint))}, {controller.num_steps} "
        f"evaluations, strength {float(strength):.2f}."
    )
    return controller


def release_pdd_parallel_heads(transformer: nn.Module) -> None:
    """Restore Maestro's ordinary output heads after a PDD generation."""

    final_layer = getattr(transformer, "final_layer", None)
    if final_layer is None:
        return
    video = getattr(final_layer, "video_out", None)
    audio = getattr(final_layer, "audio_out", None)
    if isinstance(video, MiniMaxH3PDDParallelHead):
        final_layer.video_out = video.base
    if isinstance(audio, MiniMaxH3PDDParallelHead):
        final_layer.audio_out = audio.base
    if hasattr(transformer, "_pdd_controller"):
        delattr(transformer, "_pdd_controller")


__all__ = [
    "PDD_AUDIO_SHIFT",
    "PDD_BLOCK_SIZE",
    "PDD_NUM_EVALUATIONS",
    "PDD_NUM_INTERVALS",
    "PDD_VIDEO_SHIFT",
    "MiniMaxH3PDDController",
    "MiniMaxH3PDDParallelHead",
    "install_pdd_parallel_heads",
    "is_pdd_state_dict",
    "pdd_sampling_plan",
    "pdd_sampling_plans_for_sigmas",
    "pdd_time_grid",
    "preprocess_pdd_lora_state_dict",
    "release_pdd_parallel_heads",
]
