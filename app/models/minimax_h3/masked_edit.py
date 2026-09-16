"""H3 grouped-row mask conditioning and latent-aligned outpaint geometry.

Adapted from Wan2GP v12.71; grouping keeps protected rows near clean while
editable rows follow the denoising schedule. Original spatial positions are
retained in RoPE and predictions are restored to their original row order.
"""
import math
import torch
import torch.nn.functional as F


def outpaint_location(height, width, margins):
    if margins is None:
        return None
    if not isinstance(margins, (list, tuple)) or len(margins) != 4:
        raise ValueError("H3 outpaint requires top, bottom, left and right margins")
    margins = [float(value) for value in margins]
    if any(not math.isfinite(value) or value < 0 for value in margins):
        raise ValueError("H3 outpaint margins must be finite and nonnegative")
    if not any(margins):
        return None
    from shared.utils.utils import get_outpainting_frame_location
    return get_outpainting_frame_location(height, width, margins, 1, quantize_margins=32)


def outpaint_mask(video, margins):
    location = outpaint_location(*video.shape[-2:], margins)
    if location is None:
        return None
    height, width, top, left = location
    mask = torch.ones((1, video.shape[1], *video.shape[-2:]), dtype=torch.float32, device=video.device)
    mask[:, :, top:top + height, left:left + width] = 0
    return mask


def snap_mask_to_patches(mask, patch_size):
    # A transformer token must have one noise level. If any latent cell in
    # the patch is editable, make the complete patch editable.
    return F.max_pool3d(mask, kernel_size=patch_size, stride=patch_size).repeat_interleave(
        patch_size[0], 2).repeat_interleave(patch_size[1], 3).repeat_interleave(patch_size[2], 4)


def grouped_rows(mask_rows):
    editable = mask_rows.bool().any(dim=-1)
    fixed = (~editable).nonzero().flatten()
    order = torch.cat([fixed, editable.nonzero().flatten()])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return order, inverse, fixed.numel()


def grouped_timesteps(times, indices, layout, fixed_rows):
    row_times = times[indices].clone()
    video_start = layout.num_condition_video_rows
    row_times[layout.video_indices[video_start:video_start + fixed_rows].to(row_times.device)] = 0.999
    return torch.unique(row_times, sorted=True, return_inverse=True)
