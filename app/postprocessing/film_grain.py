# Thanks to https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/film_grain.py
import torch

def add_film_grain(images: torch.Tensor, grain_intensity: float = 0, saturation: float = 0.5):
    """Apply film grain effect using GPU acceleration with chunked processing.

    Input: [C, T, H, W] tensor (float or uint8), on any device.
    Output: same shape, dtype, and device as input.
    """
    if grain_intensity <= 0:
        return images

    orig_device = images.device
    input_was_uint8 = images.dtype == torch.uint8

    # Move to GPU for fast processing
    device = torch.device('cuda') if torch.cuda.is_available() else orig_device

    # Process in chunks of frames to limit peak VRAM usage
    # For a 1280x720 frame: ~3.5MB per frame in float32 x3 channels, so 32 frames ≈ 340MB
    CHUNK_SIZE = 32
    num_frames = images.shape[1]
    output_chunks = []

    for start in range(0, num_frames, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, num_frames)
        chunk = images[:, start:end].to(device)

        if input_was_uint8:
            chunk = chunk.float().div_(255.0).mul_(2.0).sub_(1.0)

        # [C, T, H, W] -> [T, H, W, C]
        chunk = chunk.permute(1, 2, 3, 0)
        chunk.add_(1.0).div_(2.0)

        # Generate grain directly in the right shape
        grain = torch.randn(chunk.shape, device=device)
        grain[:, :, :, 0] *= 2
        grain[:, :, :, 2] *= 3
        luma = grain[:, :, :, 1:2]  # [T, H, W, 1] - no copy
        grain = grain * saturation + luma * (1 - saturation)

        # Blend
        chunk.add_(grain, alpha=grain_intensity)
        chunk.clamp_(0, 1)
        chunk.sub_(0.5).mul_(2.0)

        # [T, H, W, C] -> [C, T, H, W]
        chunk = chunk.permute(3, 0, 1, 2)

        if input_was_uint8:
            chunk = chunk.add(1.0).mul_(127.5).clamp_(0, 255).to(torch.uint8)

        output_chunks.append(chunk.to(orig_device))
        del chunk, grain

    return torch.cat(output_chunks, dim=1)
