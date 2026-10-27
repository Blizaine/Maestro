# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Fibo's FLUX.2 VAE adapter using Maestro's existing FLUX.2 autoencoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from models.flux.modules.autoencoder_flux2 import AutoEncoderParamsFlux2, AutoencoderKLFlux2
from .progress import GenerationAborted, check_abort


def _encode_tile_starts(length: int, tile_size: int) -> list[int]:
    count = max(1, math.ceil(length / tile_size))
    if count == 1:
        return [0]
    last = length - tile_size
    intervals = count - 1
    return [int(round((last * index / intervals) / 16) * 16) for index in range(count)]


def _decode_tile_starts(length: int, tile_size: int) -> list[int]:
    stride = max(1, tile_size * 3 // 4)
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] + tile_size < length:
        starts.append(length - tile_size)
    return starts


class FiboVAE(AutoencoderKLFlux2):
    """Use normalized 128-channel FLUX.2 latents and overlap tiles for Fibo."""

    def __init__(self):
        super().__init__(AutoEncoderParamsFlux2(resolution=1024))
        self.bn._lock_dtype = torch.float32
        self.encoder._lock_dtype = torch.float16
        self.decoder._lock_dtype = torch.float16
        self._convertWeightsFloatTo = None
        self._interrupt = False
        # MMGP wraps this top-level encode hook; decoder inference goes through
        # normal forward so that it uses the same current placement boundary.
        self._offload_hooks = ["encode"]

    def normalize(self, z):
        z = z.float()
        mean = self.bn.running_mean.reshape(1, -1, 1, 1).to(z)
        scale = (self.bn.running_var + self.bn_eps).rsqrt().reshape(1, -1, 1, 1).to(z)
        return (z - mean) * scale

    @staticmethod
    def get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision):
        if int(vae_config or 0) == 0:
            vae_config = 1 if device_mem_capacity >= 16000 else 2 if device_mem_capacity >= 8000 else 3
        try:
            tile_size = {1: 1024, 2: 512, 3: 256}[int(vae_config)]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Unsupported Fibo VAE tiling preset: {vae_config!r}.") from error
        return True, tile_size

    def _check_abort(self):
        check_abort()
        if self._interrupt:
            raise GenerationAborted

    def _leaf_abort_hooks(self, module):
        def abort(*_):
            self._check_abort()

        return [
            child.register_forward_pre_hook(abort)
            for child in module.modules()
            if not child._modules
        ]

    def encode(self, pixels, tile_size=1024):
        """Encode RGB pixels to normalized, 128-channel packed latent maps."""
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("Fibo VAE expects RGB tensors shaped [batch, 3, height, width].")
        height, width = pixels.shape[-2:]
        if height < 16 or width < 16 or height % 16 or width % 16:
            raise ValueError("Fibo VAE input dimensions must be positive multiples of 16 pixels.")
        try:
            tile_size = int(tile_size or 0)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Fibo VAE encoder tile size must be a non-negative integer.") from error
        if tile_size < 0 or (tile_size and tile_size % 16):
            raise ValueError("Fibo VAE encoder tile size must be zero or a multiple of 16 pixels.")
        if tile_size == 0:
            tile_size = max(height, width)
        tile_size = min(tile_size, max(height, width))
        tile_size = max(16, tile_size // 16 * 16)
        if height <= tile_size and width <= tile_size:
            self._check_abort()
            return super().encode(pixels)

        rows = _encode_tile_starts(height, min(tile_size, height))
        cols = _encode_tile_starts(width, min(tile_size, width))
        latent_h, latent_w = height // 16, width // 16
        result = torch.zeros((pixels.shape[0], 128, latent_h, latent_w), device=pixels.device, dtype=torch.float32)
        weights = torch.zeros((1, 1, latent_h, latent_w), device=pixels.device, dtype=torch.float32)
        hooks = self._leaf_abort_hooks(self.encoder)
        try:
            for row_index, row in enumerate(rows):
                self._check_abort()
                tile_h = min(tile_size, height - row)
                previous_row_end = rows[row_index - 1] + tile_h if row_index else row
                next_row = rows[row_index + 1] if row_index + 1 < len(rows) else row + tile_h
                y0, y1 = row // 16, (row + tile_h) // 16
                ramp_h = torch.ones(y1 - y0, device=pixels.device, dtype=torch.float32)
                if row_index:
                    overlap_h = max(0, (previous_row_end - row) // 16)
                    if overlap_h:
                        ramp_h[:overlap_h] *= torch.linspace(
                            0.5 / overlap_h, 1 - 0.5 / overlap_h, overlap_h,
                            device=pixels.device,
                        )
                if row_index + 1 < len(rows):
                    overlap_h = max(0, (row + tile_h - next_row) // 16)
                    if overlap_h:
                        ramp_h[-overlap_h:] *= torch.linspace(
                            1 - 0.5 / overlap_h, 0.5 / overlap_h, overlap_h,
                            device=pixels.device,
                        )
                for col_index, col in enumerate(cols):
                    self._check_abort()
                    tile_w = min(tile_size, width - col)
                    previous_col_end = cols[col_index - 1] + tile_w if col_index else col
                    next_col = cols[col_index + 1] if col_index + 1 < len(cols) else col + tile_w
                    x0, x1 = col // 16, (col + tile_w) // 16
                    ramp_w = torch.ones(x1 - x0, device=pixels.device, dtype=torch.float32)
                    if col_index:
                        overlap_w = max(0, (previous_col_end - col) // 16)
                        if overlap_w:
                            ramp_w[:overlap_w] *= torch.linspace(
                                0.5 / overlap_w, 1 - 0.5 / overlap_w, overlap_w,
                                device=pixels.device,
                            )
                    if col_index + 1 < len(cols):
                        overlap_w = max(0, (col + tile_w - next_col) // 16)
                        if overlap_w:
                            ramp_w[-overlap_w:] *= torch.linspace(
                                1 - 0.5 / overlap_w, 0.5 / overlap_w, overlap_w,
                                device=pixels.device,
                            )
                    crop = pixels[..., row:row + tile_h, col:col + tile_w].to(self.encoder.conv_in.weight.dtype)
                    moments = self.encoder(crop)
                    if moments.ndim != 4 or moments.shape[1] != 64:
                        raise ValueError("Fibo FLUX.2 VAE encoder must return 64-channel moments.")
                    packed = F.pixel_unshuffle(moments[:, :32], 2)
                    mean = self.bn.running_mean.view(1, -1, 1, 1).to(device=pixels.device, dtype=torch.float32)
                    variance = self.bn.running_var.view(1, -1, 1, 1).to(device=pixels.device, dtype=torch.float32)
                    encoded = (packed.float() - mean) * torch.rsqrt(variance + self.bn_eps)
                    weight = ramp_h[:, None] * ramp_w[None, :]
                    result[..., y0:y1, x0:x1].add_(encoded * weight)
                    weights[..., y0:y1, x0:x1].add_(weight)
                    del crop, moments, packed, encoded
        finally:
            for hook in hooks:
                hook.remove()
        self._check_abort()
        return result.div_(weights.clamp_min(torch.finfo(torch.float32).tiny)).to(dtype=pixels.dtype)

    def forward(self, latents, tile_size=0):
        """Decode normalized 128-channel Fibo latents to CPU uint8 RGB."""
        if latents.ndim != 4 or latents.shape[1] != 128:
            raise ValueError("Fibo VAE decode expects [batch, 128, height, width] latents.")
        try:
            tile_size = int(tile_size or 0)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Fibo VAE decoder tile size must be a non-negative integer.") from error
        if tile_size < 0 or (tile_size and tile_size % 16):
            raise ValueError("Fibo VAE decoder tile size must be zero or a multiple of 16 pixels.")

        height, width = latents.shape[-2:]
        size = tile_size // 16 if tile_size else max(height, width)
        size = max(1, min(size, max(height, width)))
        rows, cols = _decode_tile_starts(height, min(size, height)), _decode_tile_starts(width, min(size, width))
        output_height, output_width = height * 16, width * 16
        buffer_height = min(size, height) * 16
        result = torch.zeros((latents.shape[0], 3, buffer_height, output_width), device="cpu", dtype=torch.float32)
        weights = torch.zeros((1, 1, buffer_height, output_width), device="cpu", dtype=torch.float32)
        pixels = torch.empty((latents.shape[0], 3, output_height, output_width), device="cpu", dtype=torch.uint8)
        hooks = self._leaf_abort_hooks(self.decoder)
        try:
            stride = max(1, size * 3 // 4)
            for row_index, row in enumerate(rows):
                self._check_abort()
                for col in cols:
                    self._check_abort()
                    latent_tile = latents[..., row:row + size, col:col + size]
                    decoder_input = self.pre_decode(latent_tile).to(dtype=self.decoder.conv_in.weight.dtype)
                    decoded = self.decoder(decoder_input).float().cpu()
                    tile_height, tile_width = decoded.shape[-2:]
                    overlap = max(0, (size - stride) * 16)
                    ramp_h = torch.ones(tile_height, device="cpu", dtype=torch.float32)
                    ramp_w = torch.ones(tile_width, device="cpu", dtype=torch.float32)
                    if row and overlap:
                        ramp_h[:overlap] *= torch.linspace(0, 1, overlap, device="cpu")
                    if row + size < height and overlap:
                        ramp_h[-overlap:] *= torch.linspace(1, 0, overlap, device="cpu")
                    if col and overlap:
                        ramp_w[:overlap] *= torch.linspace(0, 1, overlap, device="cpu")
                    if col + size < width and overlap:
                        ramp_w[-overlap:] *= torch.linspace(1, 0, overlap, device="cpu")
                    weight = ramp_h[:, None] * ramp_w[None, :]
                    x = col * 16
                    result[..., :tile_height, x:x + tile_width].add_(decoded * weight)
                    weights[..., :tile_height, x:x + tile_width].add_(weight)

                start_y = row * 16
                stop_y = rows[row_index + 1] * 16 if row_index + 1 < len(rows) else output_height
                count = stop_y - start_y
                ready = result[..., :count].div_(weights[..., :count].clamp_min(torch.finfo(torch.float32).tiny))
                pixels[..., start_y:stop_y, :].copy_(ready.add_(1).mul_(127.5).clamp_(0, 255))
                remaining = buffer_height - count
                if remaining:
                    result[..., :remaining, :].copy_(result[..., count:, :].clone())
                    weights[..., :remaining, :].copy_(weights[..., count:, :].clone())
                result[..., remaining:, :].zero_()
                weights[..., remaining:, :].zero_()
        finally:
            for hook in hooks:
                hook.remove()
        self._check_abort()
        return pixels
