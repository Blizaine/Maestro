# Copyright 2026 The MiniMax and Hugging Face teams.
# Copyright 2026 Maestro contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Checkpoint adapters for MiniMax H3 consumer weights.

The video VAE conversion follows Hugging Face Diffusers' official H3
conversion script pinned in ``UPSTREAM.md``.  It is performed lazily by
MMGP while the checkpoint is loaded, allowing Maestro to use Comfy-Org's
single 5.2 GB FP16 VAE instead of the 10.4 GB three-shard Diffusers export.

Comfy-Org's audio VAE stores the already-computed convolution weights while
the reference module keeps PyTorch's legacy weight-normalization parameters.
The second adapter reconstructs an equivalent ``weight_g`` / ``weight_v``
pair at load time, avoiding a duplicate 605 MB audio checkpoint.
"""

from __future__ import annotations

import torch


VIDEO_VAE_HEADS = 32
VIDEO_VAE_HEAD_DIM = 64


def _reorder_interleaved_qkv(weight: torch.Tensor) -> torch.Tensor:
    """Convert ``[head0 qkv, head1 qkv, ...]`` to ``[q_all, k_all, v_all]``."""

    expected = VIDEO_VAE_HEADS * 3 * VIDEO_VAE_HEAD_DIM
    if weight.shape[0] != expected:
        raise ValueError(
            f"MiniMax H3 video-VAE fused QKV has {weight.shape[0]} rows; expected {expected}."
        )
    grouped = weight.reshape(VIDEO_VAE_HEADS, 3 * VIDEO_VAE_HEAD_DIM, *weight.shape[1:])
    query, key, value = grouped.split(VIDEO_VAE_HEAD_DIM, dim=1)
    return torch.cat(
        [part.reshape(VIDEO_VAE_HEADS * VIDEO_VAE_HEAD_DIM, *weight.shape[1:]) for part in (query, key, value)],
        dim=0,
    )


def _rename_video_vae_key(source_key: str) -> str:
    target_key = source_key
    if target_key.startswith("encoder.down."):
        level, rest = target_key.removeprefix("encoder.down.").split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        target_key = f"encoder.down_blocks.{level}.{rest}"
    target_key = target_key.replace("decoder.x_embedder.", "decoder.proj_in.")
    target_key = target_key.replace(".attn.to_out.", ".attn.to_out.0.")
    target_key = target_key.replace(".ff.w1.", ".ff.net.0.proj.")
    target_key = target_key.replace(".ff.w2.", ".ff.net.2.")
    return target_key


def preprocess_video_vae_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map the compact Comfy-Org video VAE onto the official Diffusers module."""

    converted: dict[str, torch.Tensor] = {}
    for source_key, tensor in state_dict.items():
        # These are config values in the official module rather than state
        # tensors.  ``decoder.mask_token`` belonged only to MAE training.
        if source_key in {"decoder.mask_token", "latents_mean", "latents_std"}:
            continue

        if ".attn.to_qkv." in source_key:
            reordered = _reorder_interleaved_qkv(tensor)
            query, key, value = reordered.chunk(3, dim=0)
            prefix, suffix = source_key.split(".attn.to_qkv.")
            converted[f"{prefix}.attn.to_q.{suffix}"] = query.contiguous()
            converted[f"{prefix}.attn.to_k.{suffix}"] = key.contiguous()
            converted[f"{prefix}.attn.to_v.{suffix}"] = value.contiguous()
            continue

        target_key = _rename_video_vae_key(source_key)
        if ".ff.w1." in source_key:
            gate, value = tensor.chunk(2, dim=0)
            tensor = torch.cat([value, gate], dim=0).contiguous()
        converted[target_key] = tensor
    return converted


def preprocess_audio_vae_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Restore weight-normalization parameters in the compact audio VAE.

    For a materialized weight ``w``, choosing ``weight_v = w`` and
    ``weight_g = ||w||`` makes PyTorch's weight-normalization parametrization
    evaluate back to ``w``.  MiniMax H3 applies weight norm to every
    convolution in the audio encoder and decoder, but not to the projection
    layers that live outside those two namespaces.
    """

    converted: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key in {"latents_mean", "latents_std"}:
            continue
        if key.endswith(".weight") and key.startswith(("encoder.", "decoder.")):
            reduce_dims = tuple(range(1, tensor.ndim))
            converted[key.removesuffix(".weight") + ".weight_g"] = torch.linalg.vector_norm(
                tensor,
                ord=2,
                dim=reduce_dims,
                keepdim=True,
            )
            converted[key.removesuffix(".weight") + ".weight_v"] = tensor
            continue
        converted[key] = tensor
    return converted
