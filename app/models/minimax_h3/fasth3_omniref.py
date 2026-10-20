# SPDX-License-Identifier: Apache-2.0
"""Pinned community FastH3 OmniRef conversion and native PDD-8 recipe.

PDD arithmetic follows FastVideo at d4f144581d6882d5bceb288ca13d767c054ca041,
fastvideo/layers/pdd.py. This is distinct from the T2AV DMD schedule and the
Alibaba acceleration adapter: its fine grid ends at .999, not 1.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from torch import nn

from . import fasth3

REPO = "lrigamai/FastH3-OmniRef-Comfy"
REVISION = "0d2c00a333ab1393d3ff18a008d7d9eda5536f11"
FILE = "fasth3_omniref_8step_pruned_int8_convrot.safetensors"
SIZE = 22213723064
SHA256 = "851b2dc9670ed864b7f1e951ac9da047d83b1c04ded9bfcc1a9e5bba0ee33f80"
GRID_MAX = 0.999
HEADS = 32
EVALUATIONS = 8
HELP = (
    "FASTH3 OMNIREF — EXPERIMENTAL\n"
    "Community eight-step Ref2VA conversion with ordered image, video and audio references. "
    "Uses 32 PDD heads in eight blocks, Euler, shifts 12/3 and trained VSA-H3 "
    "attention (90% sparsity, 4×4×8 tiles). At least one image or video reference is required. "
    "Available in Studio References. Director, start/end frames, continuation, edit masks, additional LoRAs and audio refinement are unavailable. "
    "Begin at 480p / 5.2 seconds. Consumer-GPU real-time performance is unverified."
)


def verify_checkpoint(filename: str, header: dict) -> None:
    fasth3.verify_checkpoint_pin(filename, SIZE, SHA256, "FastH3 OmniRef")
    metadata = header.get("__metadata__", {})
    try:
        config = json.loads(metadata.get("config", "{}"))
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid FastH3 OmniRef metadata") from error
    if metadata.get("source") != "transformer_ref" or config.get("transformer", {}).get("pdd_steps") != HEADS:
        raise ValueError("FastH3 OmniRef requires the pinned Ref2VA 32-head PDD conversion.")
    for name, width in (("video", 96), ("audio", 32)):
        for suffix, shape in (("weight", [HEADS * width, 5376]), ("bias", [HEADS * width])):
            spec = header.get(f"final_layer.{name}_out.{suffix}", {})
            if spec.get("shape") != shape or spec.get("dtype") != "F32":
                raise ValueError(f"FastH3 OmniRef has invalid {name} PDD output heads.")


def fine_sigmas(shift: float) -> torch.Tensor:
    if shift not in (12.0, 3.0):
        raise ValueError("FastH3 OmniRef requires video/audio shifts 12/3.")
    base = torch.linspace(GRID_MAX, 0, HEADS + 1, dtype=torch.float64, device="cpu")
    return shift * base * GRID_MAX / (base * (shift - 1) + GRID_MAX)


def sigmas(shift: float) -> tuple[float, ...]:
    return tuple(fine_sigmas(shift)[::4].tolist())


def head_weights(shift: float) -> torch.Tensor:
    # Algebraic sigma differences avoid cancellation near the noisy endpoint.
    if shift not in (12.0, 3.0):
        raise ValueError("FastH3 OmniRef requires video/audio shifts 12/3.")
    base = torch.linspace(GRID_MAX, 0, HEADS + 1, dtype=torch.float64, device="cpu")
    delta = shift * GRID_MAX**2 * (base[1:] - base[:-1]) / (
        (GRID_MAX + (shift - 1) * base[:-1]) * (GRID_MAX + (shift - 1) * base[1:]))
    blocks = delta.view(EVALUATIONS, 4)
    return (blocks / blocks.sum(-1, keepdim=True)).float()


class PDDOutput(nn.Linear):
    """Mmap/offload-compatible widened weights; emit one fused FP32 head.

Keep source names and weight shapes intact for MMGP. ``out_features`` is the
runtime width so the selected-row projector allocates only one head's rows.
Fusion occurs inside forward, after MMGP has made the parameter resident.
"""
    def __init__(self, input_size: int, output_size: int, shift: float):
        super().__init__(input_size, output_size * HEADS, bias=True, dtype=torch.float32)
        self.out_features = output_size
        self._lock_dtype = torch.float32
        self.step = 0
        self._plans = head_weights(shift).tolist()

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        if not 0 <= self.step < EVALUATIONS:
            raise ValueError("Invalid FastH3 OmniRef PDD step")
        start = self.step * 4
        weights = self.weight.view(HEADS, self.out_features, self.in_features)
        biases = self.bias.view(HEADS, self.out_features)
        plan = torch.tensor(self._plans[self.step], device=weights.device, dtype=torch.float32)
        # This Comfy conversion stores H0, then (Hj-H0) for j>0, unlike
        # FastVideo's absolute head bank. The weighted mean is always H0
        # plus weighted residuals, with no residual for interval zero.
        if start == 0:
            plan[0] = 0
        fused_weight = weights[0].float() + torch.einsum("n,noi->oi", plan, weights[start:start + 4].float())
        fused_bias = biases[0].float() + torch.einsum("n,no->o", plan, biases[start:start + 4].float())
        return F.linear(rows.float(), fused_weight, fused_bias)


def validate_request(inputs: dict) -> None:
    # The shared validator owns fixed steps, cache, LoRA and window policy.
    # Remove only ordered Ref2VA conditioning while retaining all other checks.
    check = dict(inputs)
    refs = check.pop("minimax_h3_references", None) or []
    from .reference_manifest import validate_reference_manifest
    refs = validate_reference_manifest(refs, require_files=False)
    if any(ref.get("audio_intent") == "drive" for ref in refs):
        raise ValueError("FastH3 OmniRef supports generative reference audio, not an exact target soundtrack.")
    fasth3.validate_fasth3_request(check)
    inputs.update({key: value for key, value in check.items() if key not in inputs or key in (
        "num_inference_steps", "guidance_scale", "flow_shift", "audio_flow_shift", "override_attention",
        "skip_steps_cache_type", "minimax_h3_turbo_mode", "minimax_h3_turbo_preset",
        "minimax_h3_extended_duration", "multi_prompts_gen_type", "sliding_window_size",
        "sliding_window_overlap", "sliding_window_discard_last_frames")})
    inputs["flow_shift"] = 12.0
    inputs["minimax_h3_references"] = refs


def capabilities() -> dict:
    result = fasth3.fasth3_capabilities()
    result.update({"minimax_h3_fasth3_omniref": True, "omni_reference": True,
                   "supports_reference_audio": True, "minimax_h3_video_shift": 12.0,
                   "selector_help": HELP,
                   "inference_steps_help": "FastH3 OmniRef uses eight fixed PDD evaluations and trained VSA-H3 attention."})
    return result
