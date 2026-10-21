# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""MOSS-specific tensor helpers expressed with Maestro's PyTorch runtime."""

import torch


def rows_(tensor, operation):
    """Apply a row transform in place; Maestro's attention kit has no row helper."""
    result = operation(tensor, 0, tensor.shape[1])
    if result is not None and result is not tensor:
        tensor.copy_(result)
    return tensor


def rms_norm_rounded(inputs, eps, manual=False):
    x = inputs.pop()
    dtype = x.dtype
    values = x.float()
    normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    return normalized.to(dtype=dtype)


def norm_scale_shift(inputs, norm, scale, shift):
    x = inputs.pop()
    normalized = norm(x)
    return normalized * (1 + scale.to(normalized)) + shift.to(normalized)


def scale_shift(inputs, scale, shift):
    x = inputs.pop()
    return x * (1 + scale.to(x)) + shift.to(x)


def gated_residual_(x, residuals, gate):
    residual = residuals.pop()
    x.add_(residual * gate.to(residual))
    return x
