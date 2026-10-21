"""Context-aware single-image VAE tiles, accumulated in CPU memory.

Every tile reaches a real image edge or has neighboring context. Shared
decoder edges receive zero weight where padding affects their result. This
avoids exposing a narrow final tile or averaging its unreliable edge pixels
into the finished image. Geometry is measured in VAE latent coordinates.
"""
import math

import torch


MIN_TILE = 32
MIN_OVERLAP = 16
EDGE_GUARD = 4
EDGE_RAMP = 8


def tile_starts(length, tile, overlap):
    """Cover an axis with full-size tiles, flush against both outer edges."""
    if length <= 0 or tile <= 0 or overlap < 0 or overlap >= tile:
        raise ValueError("Invalid Qwen VAE tile geometry")
    tile = min(length, tile)
    if length == tile:
        return [0]
    intervals = math.ceil((length - tile) / (tile - overlap))
    return [index * (length - tile) // intervals for index in range(intervals + 1)]


def image_tile_axis(length, requested_pixels, scale, enabled=True):
    """Keep a minimum context size even for the aggressive UI preset."""
    if not enabled:
        return length, [0]
    tile = min(length, max(MIN_TILE, int(requested_pixels) // scale))
    if tile == length:
        return tile, [0]
    overlap = max(MIN_OVERLAP, tile // 4)
    return tile, tile_starts(length, tile, overlap)


def axis_weights(length, scale, before, after):
    """Downweight only artificial boundaries, preserving real image edges."""
    pixels = length * scale
    weights = torch.ones(pixels, dtype=torch.float32, device="cpu")
    distance = torch.arange(pixels, dtype=torch.float32, device="cpu") + 0.5
    guard, ramp = EDGE_GUARD * scale, EDGE_RAMP * scale
    if before:
        weights *= ((distance - guard) / ramp).clamp_(0, 1)
    if after:
        weights *= ((pixels - distance - guard) / ramp).clamp_(0, 1)
    return weights


@torch.inference_mode()
def decode_image_to_cpu(vae, z, output_channels):
    """Decode one independent image per batch item without temporal caches.

    Tiles decode sequentially; completed pixels accumulate in FP32 on CPU
    before quantizing once, including the alpha channel.
    """
    batch, _, frames, height, width = z.shape
    if frames != 1:
        raise ValueError("Qwen Image 2.1 decodes a single-frame latent tensor.")
    scale = int(vae.spatial_compression_ratio)
    tile_h, ys = image_tile_axis(height, vae.tile_sample_min_height, scale, vae.use_tiling)
    tile_w, xs = image_tile_axis(width, vae.tile_sample_min_width, scale, vae.use_tiling)
    print(
        f"[Qwen Image 2.1 VAE] Decode {width * scale}x{height * scale}: "
        f"{tile_w * scale}x{tile_h * scale} tiles, {len(xs)}x{len(ys)} layout; "
        "single-image cache off, context-weighted CPU blend."
    )
    device = z.device
    dtype = getattr(vae, "_model_dtype", next(vae.parameters()).dtype)
    source = z.detach().cpu() if z.device.type != "cpu" else z.detach()
    del z
    output = torch.empty((batch, output_channels, 1, height * scale, width * scale), dtype=torch.uint8, device="cpu")
    patch_size = vae.config.patch_size or 1
    try:
        for batch_index in range(batch):
            canvas = torch.zeros((output_channels, height * scale, width * scale), dtype=torch.float32, device="cpu")
            totals = torch.zeros((height * scale, width * scale), dtype=torch.float32, device="cpu")
            for y in ys:
                wy = axis_weights(tile_h, scale, y > 0, y + tile_h < height)
                for x in xs:
                    vae.clear_cache()
                    tile_latents = source[batch_index:batch_index + 1, :, :, y:y + tile_h, x:x + tile_w].to(device=device, dtype=dtype)
                    tile = vae.post_quant_conv(tile_latents)
                    del tile_latents
                    tile = vae.decoder(tile, first_chunk=True)
                    if patch_size != 1:
                        # Import lazily to keep the geometry helper independently testable.
                        from .autoencoder_kl_qwenimage21 import _unpatchify
                        tile = _unpatchify(tile, patch_size=patch_size)
                    if not torch.isfinite(tile).all():
                        raise RuntimeError("Qwen Image 2.1 VAE decoding produced non-finite pixels. Try the 32-bit VAE setting.")
                    pixels = tile[0, :output_channels, 0].cpu().float()
                    del tile
                    expected = (tile_h * scale, tile_w * scale)
                    if pixels.shape[-2:] != expected:
                        raise RuntimeError("Qwen Image 2.1 VAE tile dimensions do not match its compression ratio.")
                    wx = axis_weights(tile_w, scale, x > 0, x + tile_w < width)
                    weights = wy[:, None] * wx[None, :]
                    y0, x0 = y * scale, x * scale
                    canvas[:, y0:y0 + expected[0], x0:x0 + expected[1]].add_(pixels.mul_(weights))
                    totals[y0:y0 + expected[0], x0:x0 + expected[1]].add_(weights)
                    del pixels, weights
            if not torch.all(totals > 0):
                raise RuntimeError("Qwen Image 2.1 VAE tiles left pixels without valid decoder context.")
            canvas.div_(totals).clamp_(-1, 1).add_(1).mul_(127.5).round_()
            output[batch_index, :, 0].copy_(canvas.to(torch.uint8))
        return output
    finally:
        vae.clear_cache()
