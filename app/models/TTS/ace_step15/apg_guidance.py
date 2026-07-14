"""Adaptive Projected Guidance for ACE-Step 1.5 CFG-based variants.

Ported verbatim (minus the unused ADG/MLX helpers) from the reference
implementation shipped with the CFG-capable checkpoints, e.g.
https://huggingface.co/ACE-Step/acestep-v15-xl-sft/blob/main/apg_guidance.py

Turbo variants are guidance-distilled and never call into this module;
the base/sft variants run classifier-free guidance through
``apg_forward`` with a shared per-generation ``MomentumBuffer``
(momentum -0.75, norm threshold 2.5, dims=[1] — matching upstream
``generate_audio``).
"""

import torch


class MomentumBuffer:

    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(
    v0: torch.Tensor,  # [B, C, T]
    v1: torch.Tensor,  # [B, C, T]
    dims=[-1],
):
    dtype = v0.dtype
    device_type = v0.device.type
    if device_type == "mps":
        v0, v1 = v0.cpu(), v1.cpu()

    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype).to(device_type), v0_orthogonal.to(dtype).to(device_type)


def apg_forward(
    pred_cond: torch.Tensor,  # [B, C, T]
    pred_uncond: torch.Tensor,  # [B, C, T]
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
    dims=[-1],
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    diff_parallel, diff_orthogonal = project(diff, pred_cond, dims)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided


def cfg_forward(cond_output, uncond_output, cfg_strength):
    return uncond_output + cfg_strength * (cond_output - uncond_output)
