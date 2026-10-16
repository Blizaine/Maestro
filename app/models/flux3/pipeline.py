"""Packing and reference-image preparation from the BFL FLUX 3 reference.

The model execution lifecycle lives in ``runtime.py`` so MMGP remains the
owner of component placement.
"""

import math

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor

from .autoencoder import DOWNSAMPLING_RATIO, latent_ids
from .dit import IN_CHANNELS, MODES, Packed


IMAGE_ROPE_SCALE = 0.5
MATCH_MAX_PIXELS = 4 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 64, 4096
MAX_REFERENCES = 10
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024


def pack(samples: list[list[tuple[Tensor, Tensor] | None]]) -> Packed:
    """Pack samples as [text, image_cond or None, image] in mode-major order."""
    if not samples:
        raise ValueError("at least one sample is required")
    data = {mode: [] for mode in MODES}
    ids = {mode: [] for mode in MODES}
    mode_indices = {mode: [] for mode in MODES}
    counts = dict.fromkeys(MODES, 0)
    for text, cond, image in samples:
        if text is None or image is None:
            raise ValueError("each sample requires text and image tokens")
        for mode, item in (("text", text), ("image_cond", cond), ("image", image)):
            start = counts[mode]
            if item is not None:
                if item[0].ndim != 2 or item[1].ndim != 2 or item[1].shape != (item[0].shape[0], 4):
                    raise ValueError(f"invalid {mode} token/id shapes")
                data[mode].append(item[0])
                ids[mode].append(item[1])
                counts[mode] += item[0].shape[0]
            mode_indices[mode].append((start, counts[mode]))

    device = samples[0][0][0].device

    def concatenate(parts, width, dtype):
        return torch.cat(parts) if parts else torch.zeros((0, width), device=device, dtype=dtype)

    data = {mode: concatenate(data[mode], IN_CHANNELS[mode], torch.bfloat16) for mode in MODES}
    ids = {mode: concatenate(ids[mode], 4, torch.float32) for mode in MODES}
    timesteps = {
        mode: torch.zeros(data[mode].shape[0], device=device, dtype=torch.bfloat16)
        for mode in MODES
    }
    return Packed(data=data, ids=ids, timesteps=timesteps, mode_indices=mode_indices)


def scaled_ids(ids: Tensor) -> Tensor:
    """Apply the reference RoPE spatial scale while leaving time and layer IDs intact."""
    ids = ids.float().clone()
    ids[:, 1:3] *= IMAGE_ROPE_SCALE
    return ids


def _ceil16(value: int) -> int:
    return -(-value // DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO


def reference_canvas(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the fitted single-reference size and its centered 16px canvas."""
    if width <= 0 or height <= 0:
        raise ValueError("reference image dimensions must be positive")
    scale = min(1.0, math.sqrt(MATCH_MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, _ceil16(value))) for value in fit)
        if canvas[0] * canvas[1] <= MATCH_MAX_PIXELS:
            return fit, canvas
        scale *= 0.995


def prepare_reference(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """Fit one reference under 4 MP and reflect-pad it to a 16px canvas."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (width, height) = reference_canvas(image.width, image.height)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (width, height) == image.size:
        return image, None
    left, top = (width - image.width) // 2, (height - image.height) // 2
    padding = ((top, height - image.height - top), (left, width - image.width - left), (0, 0))
    mode = "reflect" if min(image.size) > DOWNSAMPLING_RATIO else "symmetric"
    padded = Image.fromarray(np.pad(np.asarray(image), padding, mode=mode))
    return padded, (left, top, left + image.width, top + image.height)


def prepare_multi_reference(image: Image.Image) -> Image.Image:
    """Fit a member of a multi-reference edit under 1 MP and crop to the 16px grid."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > MULTI_REFERENCE_MAX_PIXELS:
        scale = math.sqrt(MULTI_REFERENCE_MAX_PIXELS / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    width = width // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO
    height = height // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO
    if width < DOWNSAMPLING_RATIO or height < DOWNSAMPLING_RATIO:
        raise ValueError("reference images must be at least 16 pixels wide and high")
    left, top = (image.width - width) // 2, (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def to_tensor(image: Image.Image) -> Tensor:
    """Convert RGB pixels to channel-first float pixels in [-1, 1]."""
    pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()
    return pixels / 127.5 - 1.0


def target_ids(height: int, width: int, device: torch.device | str) -> Tensor:
    """Return target latent IDs with FLUX 3's spatial RoPE scale applied."""
    rows, cols = height // DOWNSAMPLING_RATIO, width // DOWNSAMPLING_RATIO
    return scaled_ids(latent_ids(rows, cols, device))
