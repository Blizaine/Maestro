"""Pinned MiniMax H3 learned x2 VAE assets and small output helpers.

This module intentionally has no import-time torch dependency so WanGP can
inspect the capability without initializing a model runtime.
"""

from __future__ import annotations

import hashlib
import os


X2_VAE_METHOD = "h3_vae"
X2_VAE_VALUE = "h3_vae*2"
X2_VAE_REPO = "DeepBeepMeep/MiniMax-H3"
X2_VAE_REVISION = "adc81ccb71352192214d83d5fafb9487e860be39"
X2_VAE_FILE = "MiniMax-H3-X2-Detail-v1_int8_convrot.safetensors"
X2_VAE_LICENSE_FILE = "MiniMax-H3-X2-Detail-v1.LICENSE"
X2_VAE_NOTICE_FILE = "MiniMax-H3-X2-Detail-v1.NOTICE"
X2_VAE_SIZE = 2_834_844_787
X2_VAE_SHA256 = "7ca1a1ad298e06e430da5c18bf721af4e0dcdf7f4ec7283391520bd0b6df7387"


def verify_x2_vae_checkpoint(filename: str) -> None:
    """Fail closed when a cached x2 checkpoint is incomplete or not pinned."""

    try:
        size = os.path.getsize(filename)
    except OSError as error:
        raise FileNotFoundError(f"MiniMax H3 x2 VAE checkpoint is unavailable: {filename}") from error
    if size != X2_VAE_SIZE:
        raise ValueError(
            "MiniMax H3 x2 VAE checkpoint has the wrong size "
            f"({size:,} bytes; expected {X2_VAE_SIZE:,}). Re-download "
            f"{X2_VAE_REPO}@{X2_VAE_REVISION}/{X2_VAE_FILE}."
        )

    digest = hashlib.sha256()
    with open(filename, "rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != X2_VAE_SHA256:
        raise ValueError(
            "MiniMax H3 x2 VAE checkpoint failed SHA-256 verification "
            f"(got {actual}; expected {X2_VAE_SHA256}). Re-download "
            f"{X2_VAE_REPO}@{X2_VAE_REVISION}/{X2_VAE_FILE}."
        )


def resize_video_canvas_uint8(video, target_height: int, target_width: int):
    """Resize a CPU ``[C,F,H,W]`` video directly into compact uint8 frames.

    Float inputs use the H3/WanGP normalized ``[-1, 1]`` convention. Integer
    inputs are interpreted as pixels in ``[0, 255]``. Resizing and byte
    conversion are bounded to two frames at a time.
    """

    import torch
    import torch.nn.functional as F

    if not torch.is_tensor(video) or video.ndim != 4:
        raise ValueError("H3 video must have shape [C,F,H,W].")
    target_height, target_width = int(target_height), int(target_width)
    if target_height <= 0 or target_width <= 0:
        raise ValueError("H3 video output dimensions must be positive.")
    video = video.detach().to("cpu")
    channels, frames, height, width = video.shape
    if video.dtype not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f"Unsupported H3 video dtype: {video.dtype}.")
    output = torch.empty((channels, frames, target_height, target_width), dtype=torch.uint8, device="cpu")
    for start in range(0, frames, 2):
        stop = min(start + 2, frames)
        # With a single frame, the permuted tensor can already be considered
        # contiguous; for float32, to() can then alias the caller's buffer.
        # Clone this two-frame working set before any in-place conversion.
        frame_batch = video[:, start:stop].permute(1, 0, 2, 3).contiguous().to(torch.float32).clone()
        if (height, width) != (target_height, target_width):
            frame_batch = F.interpolate(
                frame_batch,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        if video.dtype == torch.uint8:
            frame_batch = frame_batch.round_().clamp_(0, 255).to(torch.uint8)
        else:
            frame_batch.clamp_(-1.0, 1.0).add_(1.0).mul_(127.5).clamp_(0, 255)
            frame_batch = frame_batch.to(torch.uint8)
        output[:, start:stop].copy_(frame_batch.permute(1, 0, 2, 3))
    return output


def resize_video_canvas(video, target_height: int, target_width: int):
    """Resize a CPU ``[C,F,H,W]`` prefix while preserving its dtype and range.

    Frames are processed in batches of at most eight so a large x2 continuation
    prefix does not create a second full-resolution floating-point video.
    Float inputs are assumed to already be normalized; uint8 inputs stay in
    ``[0,255]``.
    """

    import torch
    import torch.nn.functional as F

    if not torch.is_tensor(video) or video.ndim != 4:
        raise ValueError("H3 continuation video must have shape [C,F,H,W].")
    target_height, target_width = int(target_height), int(target_width)
    if target_height <= 0 or target_width <= 0:
        raise ValueError("H3 continuation output dimensions must be positive.")
    video = video.detach().to("cpu")
    channels, frames, height, width = video.shape
    if (height, width) == (target_height, target_width):
        return video.contiguous()
    if video.dtype not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f"Unsupported H3 continuation dtype: {video.dtype}.")

    resized = torch.empty(
        (channels, frames, target_height, target_width),
        dtype=video.dtype,
        device="cpu",
    )
    for start in range(0, frames, 8):
        stop = min(start + 8, frames)
        frame_batch = video[:, start:stop].permute(1, 0, 2, 3).contiguous()
        frame_batch = frame_batch.to(torch.float32)
        frame_batch = F.interpolate(
            frame_batch,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        if video.dtype == torch.uint8:
            frame_batch = frame_batch.round_().clamp_(0, 255).to(torch.uint8)
        else:
            frame_batch = frame_batch.to(video.dtype)
        resized[:, start:stop].copy_(frame_batch.permute(1, 0, 2, 3))
    return resized


__all__ = [
    "X2_VAE_FILE",
    "X2_VAE_LICENSE_FILE",
    "X2_VAE_METHOD",
    "X2_VAE_NOTICE_FILE",
    "X2_VAE_REPO",
    "X2_VAE_REVISION",
    "X2_VAE_SHA256",
    "X2_VAE_SIZE",
    "X2_VAE_VALUE",
    "resize_video_canvas",
    "resize_video_canvas_uint8",
    "verify_x2_vae_checkpoint",
]
