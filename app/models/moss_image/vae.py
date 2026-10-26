# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""MOSS-Image's Flux2 small decoder with native overlapping VAE tiles."""

import math

import torch
from torch import nn

from models.flux.modules.autoencoder_flux2 import Decoder, Encoder
from shared.utils.conv_bands import band_convs

from .progress import GenerationAborted


class MOSSImageVAE(nn.Module):
    def __init__(self, encode=False):
        super().__init__()
        # The edit encoder is the pinned Flux2 full VAE encoder. It is loaded
        # from its own checkpoint so the small MOSS decoder and BN statistics
        # continue to come from the existing decoder asset.
        if encode:
            self.encoder = Encoder(
                ch=128,
                ch_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                in_channels=3,
                resolution=1024,
                z_channels=32,
            )
        # These names and shapes match the released
        # flux2_small_decoder_bf16.safetensors checkpoint.
        self.decoder = Decoder(
            ch=96,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            in_channels=3,
            resolution=1024,
            z_channels=32,
        )
        self.bn_eps = 1e-4
        self.ps = [2, 2]
        self.bn = nn.BatchNorm2d(128, eps=self.bn_eps, affine=False)
        self.bn._lock_dtype = torch.float32
        self._convertWeightsFloatTo = None
        self._interrupt = False
        self._offload_hooks = ["encode"] if encode else []
        band_convs(self)

    @staticmethod
    def get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision):
        """Choose the same memory-tiered tile sizes as Maestro's Flux3 VAE."""
        if int(vae_config or 0) == 0:
            vae_config = 1 if device_mem_capacity >= 16000 else 2 if device_mem_capacity >= 8000 else 3
        try:
            tile_size = {1: 1024, 2: 512, 3: 256}[int(vae_config)]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Unsupported MOSS Image VAE tiling preset: {vae_config!r}.") from error
        return True, tile_size

    @staticmethod
    def _tile_starts(length, size):
        stride = max(1, size * 3 // 4)
        starts = list(range(0, max(length - size, 0) + 1, stride))
        if starts[-1] + size < length:
            starts.append(length - size)
        return starts

    def _unnormalize(self, latents):
        self.bn.eval()
        mean = self.bn.running_mean.view(1, -1, 1, 1).to(latents)
        scale = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1).to(latents) + self.bn_eps)
        return latents * scale + mean

    @staticmethod
    def _unpatchify(latents):
        batch, channels, height, width = latents.shape
        if channels != 128:
            raise ValueError(f"MOSS-Image VAE expects 128-channel latents; received {channels}.")
        return (
            latents.reshape(batch, 32, 2, 2, height, width)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(batch, 32, height * 2, width * 2)
        )

    @staticmethod
    def _patch_mean(moments):
        """Take the Gaussian mean and pack 2x2 spatial cells into 128 channels."""
        if moments.ndim != 4 or moments.shape[1] != 64:
            raise ValueError("MOSS Image VAE encoder must return 64-channel moments.")
        mean = moments[:, :32]
        if mean.shape[-2] % 2 or mean.shape[-1] % 2:
            raise ValueError("MOSS Image VAE encoder dimensions must be even before latent packing.")
        return torch.nn.functional.pixel_unshuffle(mean, 2)

    @staticmethod
    def _encode_tile_starts(length, tile_size):
        count = max(1, math.ceil(length / tile_size))
        if count == 1:
            return [0]
        last = length - tile_size
        intervals = count - 1
        return [
            int(round((last * index / intervals) / 16) * 16)
            for index in range(count)
        ]

    def encode(self, pixels, tile_size=384):
        """Encode RGB references with bounded, multiple-of-16 input tiles.

        Tile crops use real source pixels, never synthetic padding. Each tile
        starts on the VAE's 16-pixel lattice and overlapping normalized latents
        are blended on bounded spatial tensors on the VAE's active device. A
        reference no larger than the configured tile is encoded in one pass.
        """
        if not hasattr(self, "encoder"):
            raise RuntimeError("MOSS Image Edit requires the full Flux2 VAE encoder.")
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("MOSS Image VAE encoder expects RGB tensors shaped [batch, 3, height, width].")
        height, width = pixels.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError("MOSS Image Edit references must be fitted to multiples of 16 pixels before encoding.")
        try:
            tile_size = int(tile_size)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("MOSS Image encoder tile size must be a positive multiple of 16 pixels.") from error
        if tile_size < 16 or tile_size % 16:
            raise ValueError("MOSS Image encoder tile size must be a positive multiple of 16 pixels.")

        tile_size = min(tile_size, max(height, width))
        tile_size = max(16, (tile_size // 16) * 16)
        if height > tile_size or width > tile_size:
            print(
                f"[MOSS Image Edit] VAE encoder uses {tile_size}px reference tiles; "
                "local mid-attention and overlap blending can change reference features."
            )
        rows = self._encode_tile_starts(height, min(tile_size, height))
        cols = self._encode_tile_starts(width, min(tile_size, width))
        dtype = self.encoder.conv_in.weight.dtype
        device = pixels.device
        latent_h, latent_w = height // 16, width // 16
        result = torch.zeros((pixels.shape[0], 128, latent_h, latent_w), device=device, dtype=torch.float32)
        weights = torch.zeros((1, 1, latent_h, latent_w), device=device, dtype=torch.float32)

        def check_abort(*_):
            if self._interrupt:
                raise GenerationAborted

        leaf_hooks = [
            module.register_forward_pre_hook(check_abort)
            for module in self.encoder.modules()
            if not module._modules
        ]
        try:
            for row_index, row in enumerate(rows):
                check_abort()
                tile_h = min(tile_size, height - row)
                previous_row_end = rows[row_index - 1] + tile_h if row_index else row
                next_row = rows[row_index + 1] if row_index + 1 < len(rows) else row + tile_h
                y0, y1 = row // 16, (row + tile_h) // 16
                ramp_h = torch.ones(y1 - y0, device=device, dtype=torch.float32)
                if row_index:
                    overlap_h = max(0, (previous_row_end - row) // 16)
                    if overlap_h:
                        ramp_h[:overlap_h] *= torch.linspace(
                            0.5 / overlap_h, 1 - 0.5 / overlap_h, overlap_h, device=device
                        )
                if row_index + 1 < len(rows):
                    overlap_h = max(0, (row + tile_h - next_row) // 16)
                    if overlap_h:
                        ramp_h[-overlap_h:] *= torch.linspace(
                            1 - 0.5 / overlap_h, 0.5 / overlap_h, overlap_h, device=device
                        )

                for col_index, col in enumerate(cols):
                    check_abort()
                    tile_w = min(tile_size, width - col)
                    previous_col_end = cols[col_index - 1] + tile_w if col_index else col
                    next_col = cols[col_index + 1] if col_index + 1 < len(cols) else col + tile_w
                    x0, x1 = col // 16, (col + tile_w) // 16
                    ramp_w = torch.ones(x1 - x0, device=device, dtype=torch.float32)
                    if col_index:
                        overlap_w = max(0, (previous_col_end - col) // 16)
                        if overlap_w:
                            ramp_w[:overlap_w] *= torch.linspace(
                                0.5 / overlap_w, 1 - 0.5 / overlap_w, overlap_w, device=device
                            )
                    if col_index + 1 < len(cols):
                        overlap_w = max(0, (col + tile_w - next_col) // 16)
                        if overlap_w:
                            ramp_w[-overlap_w:] *= torch.linspace(
                                1 - 0.5 / overlap_w, 0.5 / overlap_w, overlap_w, device=device
                            )

                    crop = pixels[..., row:row + tile_h, col:col + tile_w].to(dtype=dtype)
                    moments = self.encoder(crop)
                    packed = self._patch_mean(moments)
                    mean = self.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=torch.float32)
                    variance = self.bn.running_var.view(1, -1, 1, 1).to(device=device, dtype=torch.float32)
                    encoded = (packed.float() - mean) * torch.rsqrt(variance + self.bn_eps)
                    weight = ramp_h[:, None] * ramp_w[None, :]
                    result[..., y0:y1, x0:x1].add_(encoded.float() * weight)
                    weights[..., y0:y1, x0:x1].add_(weight)
                    del crop, moments, packed, encoded
        finally:
            for hook in leaf_hooks:
                hook.remove()
        check_abort()
        result.div_(weights.clamp_min(torch.finfo(torch.float32).tiny))
        return result.to(dtype=pixels.dtype)

    def forward(self, latents, tile_size=0):
        """Decode normalized latents to CPU uint8 RGB, limiting VRAM to one tile.

        ``tile_size`` uses output pixel units, matching Maestro's Flux3 VAE API.
        Overlap and ramp blending follow the pinned WanGP Flux3 VAE implementation.
        """
        if latents.ndim != 4:
            raise ValueError("MOSS-Image VAE expects latents with shape [batch, 128, height, width].")
        if latents.shape[1] != 128:
            raise ValueError(f"MOSS-Image VAE expects 128-channel latents; received {latents.shape[1]}.")
        try:
            tile_size = int(tile_size or 0)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("MOSS-Image VAE tile size must be a non-negative integer in pixels.") from error
        if tile_size < 0:
            raise ValueError("MOSS-Image VAE tile size must be a non-negative integer in pixels.")
        if tile_size and tile_size % 16:
            raise ValueError("MOSS-Image VAE tile size must be a multiple of 16 pixels.")

        height, width = latents.shape[-2:]
        size = tile_size // 16 if tile_size else max(height, width)
        if size <= 0:
            size = max(height, width)
        rows, cols = self._tile_starts(height, size), self._tile_starts(width, size)
        out_height, out_width = height * 16, width * 16
        buffer_height = min(size, height) * 16
        result = torch.zeros((latents.shape[0], 3, buffer_height, out_width), device="cpu", dtype=torch.float32)
        weights = torch.zeros((1, 1, buffer_height, out_width), device="cpu", dtype=torch.float32)
        pixels = torch.empty((latents.shape[0], 3, out_height, out_width), device="cpu", dtype=torch.uint8)

        def check_abort(*_):
            if self._interrupt:
                raise GenerationAborted

        leaf_hooks = [
            module.register_forward_pre_hook(check_abort)
            for module in self.decoder.modules()
            if not module._modules
        ]
        try:
            decoder_dtype = self.decoder.conv_in.weight.dtype
            for row_index, row in enumerate(rows):
                check_abort()
                for col in cols:
                    check_abort()
                    latent_tile = latents[..., row:row + size, col:col + size]
                    decoder_input = self._unpatchify(self._unnormalize(latent_tile)).to(dtype=decoder_dtype)
                    decoded = self.decoder(decoder_input).float().cpu()
                    tile_height, tile_width = decoded.shape[-2:]

                    overlap = max(0, (size - max(1, size * 3 // 4)) * 16)
                    ramp_h = torch.ones(tile_height, device="cpu")
                    ramp_w = torch.ones(tile_width, device="cpu")
                    if row and overlap:
                        ramp_h[:overlap] *= torch.linspace(0, 1, overlap, device="cpu")
                    if row + size < height and overlap:
                        ramp_h[-overlap:] *= torch.linspace(1, 0, overlap, device="cpu")
                    if col and overlap:
                        ramp_w[:overlap] *= torch.linspace(0, 1, overlap, device="cpu")
                    if col + size < width and overlap:
                        ramp_w[-overlap:] *= torch.linspace(1, 0, overlap, device="cpu")

                    weight = ramp_h[:, None] * ramp_w[None, :]
                    # Only one tile row lives in this staging buffer. Its
                    # vertical origin stays zero while completed rows roll out.
                    x = col * 16
                    result[..., :tile_height, x:x + tile_width].add_(decoded * weight)
                    weights[..., :tile_height, x:x + tile_width].add_(weight)

                # Completed output rows are emitted as bytes and removed from
                # the floating-point staging buffer before decoding continues.
                y = row * 16
                stop = rows[row_index + 1] * 16 if row_index + 1 < len(rows) else out_height
                count = stop - y
                ready = result[..., :count, :].div_(weights[..., :count, :])
                pixels[..., y:stop, :].copy_(ready.add_(1).mul_(127.5).clamp_(0, 255))
                remaining = buffer_height - count
                if remaining:
                    result[..., :remaining, :].copy_(result[..., count:, :].clone())
                    weights[..., :remaining, :].copy_(weights[..., count:, :].clone())
                result[..., remaining:, :].zero_()
                weights[..., remaining:, :].zero_()
        finally:
            for hook in leaf_hooks:
                hook.remove()
        check_abort()
        return pixels
