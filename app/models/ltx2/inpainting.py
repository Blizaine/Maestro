"""Mask preparation and boundary blending for LTX-2.3 editing.

Adapted from the published LTX-2.3 in/outpainting workflows. A value of 1 in
a mask is generated content; 0 is the protected source. Maestro's default
Outpaint path follows the current Lightricks green-marker, full-reference,
decoded-pixel workflow. Neutral/source-attention helpers remain available as
a compatibility fallback and for regression comparison.
"""

import math

import torch
import torch.nn.functional as F

from .ltx2_runtime import (
    LTX2_INPAINT_CONTROL_VIDEO_PAD_RGB,
    LTX2_LAPLACIAN_BLEND_MASK_LOW_RES_LONG_SIDE,
    LTX2_MASKED_CONTROL_VIDEO_PAD_RGB,
    LTX2_OUTPAINTING_CANVAS_MATCH,
    LTX2_OUTPAINTING_CANVAS_MATCH_MAX_CHROMA_SHIFT,
    LTX2_OUTPAINTING_CANVAS_MATCH_MAX_GAIN,
    LTX2_OUTPAINTING_CANVAS_MATCH_MAX_GRADIENT_CHROMA_SHIFT,
    LTX2_OUTPAINTING_CANVAS_MATCH_MAX_OFFSET,
    LTX2_OUTPAINTING_CANVAS_MATCH_MAX_SHARPEN,
)


def _normalize_outpainting_dims(outpainting_dims) -> list[float] | None:
    if outpainting_dims is None:
        return None
    if isinstance(outpainting_dims, str):
        outpainting_dims = outpainting_dims.strip()
        if not outpainting_dims or outpainting_dims.startswith("#"):
            return None
        outpainting_dims = outpainting_dims.split()
    if (
        not isinstance(outpainting_dims, (list, tuple))
        or len(outpainting_dims) != 4
    ):
        return None
    dims = [max(0.0, float(value)) for value in outpainting_dims]
    return dims if any(dims) else None


def _get_outpainting_inner_rect(
    height: int,
    width: int,
    outpainting_dims,
) -> tuple[int, int, int, int] | None:
    dims = _normalize_outpainting_dims(outpainting_dims)
    if dims is None or height <= 0 or width <= 0:
        return None
    from shared.utils.utils import get_outpainting_frame_location

    inner_height, inner_width, margin_top, margin_left = (
        get_outpainting_frame_location(
            int(height),
            int(width),
            dims,
            1,
        )
    )
    top = max(0, min(int(margin_top), int(height)))
    left = max(0, min(int(margin_left), int(width)))
    bottom = max(top, min(top + int(inner_height), int(height)))
    right = max(left, min(left + int(inner_width), int(width)))
    if bottom <= top or right <= left:
        return None
    return top, bottom, left, right


def _build_outpainting_mask_cthw(
    video_tensor: torch.Tensor | None,
    outpainting_dims,
) -> torch.Tensor | None:
    """Return a CTHW mask: generated canvas=1, protected source=0."""
    if (
        video_tensor is None
        or not torch.is_tensor(video_tensor)
        or video_tensor.dim() != 4
    ):
        return None
    rect = _get_outpainting_inner_rect(
        video_tensor.shape[-2],
        video_tensor.shape[-1],
        outpainting_dims,
    )
    if rect is None:
        return None
    mask = torch.ones(
        (
            1,
            video_tensor.shape[1],
            video_tensor.shape[-2],
            video_tensor.shape[-1],
        ),
        # Binary storage cuts the full-resolution video mask to one quarter
        # of the RAM used by float32. Blend chunks promote it only as needed.
        dtype=torch.uint8,
        device=video_tensor.device,
    )
    top, bottom, left, right = rect
    mask[:, :, top:bottom, left:right] = 0.0
    return mask


def _merge_ltx2_masks(
    mask: torch.Tensor | None,
    extra_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if extra_mask is None:
        return mask
    if mask is None:
        return extra_mask
    extra_mask = extra_mask.to(device=mask.device, dtype=torch.float32)
    return torch.maximum(mask.float(), extra_mask).to(dtype=mask.dtype)


def _masked_pad_color(
    video: torch.Tensor,
    rgb=LTX2_MASKED_CONTROL_VIDEO_PAD_RGB,
) -> torch.Tensor:
    if rgb is None:
        rgb = LTX2_MASKED_CONTROL_VIDEO_PAD_RGB
    color = torch.tensor(
        rgb,
        device=video.device,
        dtype=torch.float32,
    )
    if video.dtype == torch.uint8:
        return color.to(dtype=torch.uint8)
    return color.div(127.5).sub(1.0).to(dtype=video.dtype)


def _paint_ltx2_masked_control_video(
    video: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Paint generated pixels with Maestro's neutral legacy canvas."""
    if video is None or mask is None:
        return video
    color = _masked_pad_color(video).view(3, 1, 1, 1)
    return torch.where(
        mask.to(device=video.device) > 0,
        color,
        video,
    )


def _paint_ltx2_inpaint_control_video(
    video: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Paint generated pixels with Lightricks' exact ``#66FF00`` marker."""
    if video is None or mask is None:
        return video
    color = _masked_pad_color(
        video,
        LTX2_INPAINT_CONTROL_VIDEO_PAD_RGB,
    ).view(3, 1, 1, 1)
    return torch.where(
        mask.to(device=video.device) > 0,
        color,
        video,
    )


def _edge_extend_ltx2_masked_control_video(
    video: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Fill the generated canvas from the nearest protected source edge.

    This remains the single-stage fallback. The normal two-stage path uses a
    neutral canvas with source-region reference attention; edge extension
    gives the one-pass variant clean VAE boundary context as well.
    """
    if video is None or mask is None:
        return video
    if video.dim() != 4 or mask.dim() != 4:
        return video

    generated = mask.to(device=video.device) > 0
    protected = ~generated[0, 0]
    protected_rows = torch.where(protected.any(dim=1))[0]
    protected_cols = torch.where(protected.any(dim=0))[0]
    if protected_rows.numel() == 0 or protected_cols.numel() == 0:
        return video

    top = int(protected_rows[0].item())
    bottom = int(protected_rows[-1].item()) + 1
    left = int(protected_cols[0].item())
    right = int(protected_cols[-1].item()) + 1
    height, width = video.shape[-2:]
    source = video[:, :, top:bottom, left:right]
    if source.shape[-2] <= 0 or source.shape[-1] <= 0:
        return video

    # F.pad's 4-D replicate mode treats the first dimension as the batch, so
    # temporarily present frames as N and channels as C.
    extended = F.pad(
        source.permute(1, 0, 2, 3),
        (left, width - right, top, height - bottom),
        mode="replicate",
    ).permute(1, 0, 2, 3)
    return torch.where(generated.expand_as(video), extended, video)


def _pad_ltx2_masked_control_video_tail(
    video: torch.Tensor | None,
    mask: torch.Tensor | None,
    target_frames: int,
    repeat_last_frame: bool = False,
    pad_rgb=LTX2_MASKED_CONTROL_VIDEO_PAD_RGB,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if video is None:
        return video, mask
    target_frames = int(target_frames)
    video_frames = int(video.shape[1])
    if video_frames < target_frames:
        pad_frames = target_frames - video_frames
        if repeat_last_frame and video_frames > 0:
            pad = video[:, -1:].expand(
                video.shape[0],
                pad_frames,
                video.shape[-2],
                video.shape[-1],
            )
        else:
            color = _masked_pad_color(video, pad_rgb)
            pad = color.view(3, 1, 1, 1).expand(
                3,
                pad_frames,
                video.shape[-2],
                video.shape[-1],
            )
        video = torch.cat([video, pad], dim=1)
    if mask is not None and int(mask.shape[1]) < int(video.shape[1]):
        if repeat_last_frame and int(mask.shape[1]) > 0:
            pad = mask[:, -1:].expand(
                mask.shape[0],
                int(video.shape[1]) - int(mask.shape[1]),
                mask.shape[-2],
                mask.shape[-1],
            )
        else:
            pad = torch.ones(
                (
                    mask.shape[0],
                    int(video.shape[1]) - int(mask.shape[1]),
                    mask.shape[-2],
                    mask.shape[-1],
                ),
                device=mask.device,
                dtype=mask.dtype,
            )
        mask = torch.cat([mask, pad], dim=1)
    return video, mask


def _to_unit_video(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor.float().div(255.0).clamp(0.0, 1.0)
    return tensor.float().add(1.0).mul(0.5).clamp(0.0, 1.0)


def _resize_preserving_aspect_ratio(
    tensor: torch.Tensor,
    long_side: int,
    mode: str,
) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    current_long_side = max(height, width)
    if current_long_side == long_side:
        return tensor
    scale = long_side / current_long_side
    size = (
        max(1, int(round(height * scale))),
        max(1, int(round(width * scale))),
    )
    if mode == "nearest":
        return F.interpolate(tensor, size=size, mode=mode)
    return F.interpolate(
        tensor,
        size=size,
        mode=mode,
        align_corners=False,
    )


def _apply_low_res_mask_dilation(
    mask: torch.Tensor,
    spatial_radius: int,
    long_side: int = LTX2_LAPLACIAN_BLEND_MASK_LOW_RES_LONG_SIDE,
) -> torch.Tensor:
    if spatial_radius <= 0:
        return mask
    original_size = mask.shape[-2:]
    low_res = _resize_preserving_aspect_ratio(
        mask.float(),
        long_side,
        "bilinear",
    )
    low_res = F.max_pool2d(
        low_res,
        kernel_size=spatial_radius * 2 + 1,
        stride=1,
        padding=spatial_radius,
    )
    return F.interpolate(
        low_res,
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )


def _build_source_boundary_feather_alpha(
    mask: torch.Tensor,
    feather_pixels: int,
) -> torch.Tensor:
    """Limit generated influence to a narrow band inside the source rect.

    Outpaint masks are spatially static and rectangular: generated canvas
    pixels are 1 and the protected source is 0.  A full Laplacian pyramid
    carries low-frequency generated content across the entire protected
    rectangle, even when its center is hundreds of pixels from a seam.
    This alpha keeps that blend only along source edges that actually touch
    generated canvas and is exactly zero deeper inside the source.
    """
    binary = mask.detach().float().cpu().clamp(0.0, 1.0) >= 0.5
    if binary.dim() != 4 or binary.shape[0] < 1:
        return binary.float()

    feather_pixels = max(0, int(feather_pixels))
    if feather_pixels == 0:
        return binary.float()

    # The outpainting mask is identical for every frame. Build one spatial
    # alpha and broadcast it over time instead of repeating geometry work.
    generated = binary[0, 0]
    protected = ~generated
    protected_points = torch.nonzero(protected, as_tuple=False)
    if protected_points.numel() == 0:
        return torch.ones_like(binary, dtype=torch.float32)

    height, width = generated.shape
    top = int(protected_points[:, 0].min())
    bottom = int(protected_points[:, 0].max())
    left = int(protected_points[:, 1].min())
    right = int(protected_points[:, 1].max())
    region_height = bottom - top + 1
    region_width = right - left + 1
    distance = torch.full(
        (region_height, region_width),
        float(feather_pixels + 1),
        dtype=torch.float32,
        device=generated.device,
    )

    # Canvas boundaries are not seams. Only measure from an edge when
    # generated pixels exist immediately outside that side of the source.
    if top > 0 and torch.any(generated[top - 1, left : right + 1]):
        from_top = torch.arange(
            1,
            region_height + 1,
            dtype=torch.float32,
            device=generated.device,
        ).view(-1, 1)
        distance = torch.minimum(distance, from_top)
    if bottom + 1 < height and torch.any(
        generated[bottom + 1, left : right + 1]
    ):
        from_bottom = torch.arange(
            region_height,
            0,
            -1,
            dtype=torch.float32,
            device=generated.device,
        ).view(-1, 1)
        distance = torch.minimum(distance, from_bottom)
    if left > 0 and torch.any(generated[top : bottom + 1, left - 1]):
        from_left = torch.arange(
            1,
            region_width + 1,
            dtype=torch.float32,
            device=generated.device,
        ).view(1, -1)
        distance = torch.minimum(distance, from_left)
    if right + 1 < width and torch.any(
        generated[top : bottom + 1, right + 1]
    ):
        from_right = torch.arange(
            region_width,
            0,
            -1,
            dtype=torch.float32,
            device=generated.device,
        ).view(1, -1)
        distance = torch.minimum(distance, from_right)

    source_alpha = (
        (float(feather_pixels + 1) - distance)
        .div(float(feather_pixels))
        .clamp(0.0, 1.0)
    )
    spatial_alpha = generated.float()
    region_alpha = spatial_alpha[top : bottom + 1, left : right + 1]
    region_protected = protected[top : bottom + 1, left : right + 1]
    region_alpha[region_protected] = source_alpha[region_protected]
    return spatial_alpha.view(1, 1, height, width).expand(
        1,
        binary.shape[1],
        height,
        width,
    )


def _build_gaussian_margin_alpha(
    mask: torch.Tensor,
    blur_radius: int,
) -> torch.Tensor:
    """Blur a static Outpaint mask like the current Diffusers demo.

    The returned alpha is one in the generated margins and zero deep inside
    the protected source. A finite 3-sigma kernel makes the protected center
    pixel-exact while leaving a soft transition at the source boundary.
    """
    binary = mask.detach().float().cpu().clamp(0.0, 1.0)
    if binary.dim() != 4 or binary.shape[0] < 1:
        return binary
    blur_radius = max(0, int(blur_radius))
    if blur_radius == 0:
        return binary

    sigma = float(blur_radius)
    kernel_radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(
        -kernel_radius,
        kernel_radius + 1,
        dtype=torch.float32,
        device=binary.device,
    )
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    spatial = binary[:, :1]
    # MMGP sets PyTorch's global default device to CUDA while the decoded
    # postprocess deliberately runs on CPU. Keep both convolution operands
    # anchored to the mask instead of inheriting that global default.
    kernel = kernel.to(device=spatial.device, dtype=spatial.dtype)
    spatial = F.conv2d(
        F.pad(
            spatial,
            (kernel_radius, kernel_radius, 0, 0),
            mode="replicate",
        ),
        kernel.view(1, 1, 1, -1),
    )
    spatial = F.conv2d(
        F.pad(
            spatial,
            (0, 0, kernel_radius, kernel_radius),
            mode="replicate",
        ),
        kernel.view(1, 1, -1, 1),
    )
    return spatial.expand(
        1,
        binary.shape[1],
        binary.shape[-2],
        binary.shape[-1],
    )


def _laplacian_pyramid_blend(
    generated: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
    levels: int = 7,
    mask_low_res_dilation: int = 0,
) -> torch.Tensor:
    generated = _to_unit_video(generated.cpu()).permute(1, 0, 2, 3)
    source = _to_unit_video(source.cpu()).permute(1, 0, 2, 3)
    mask = mask.float().cpu().clamp(0.0, 1.0).permute(1, 0, 2, 3)
    mask = _apply_low_res_mask_dilation(
        mask,
        mask_low_res_dilation,
    ).clamp(0.0, 1.0)
    original_height, original_width = generated.shape[-2:]
    padded_height = 1 << (int(original_height) - 1).bit_length()
    padded_width = 1 << (int(original_width) - 1).bit_length()
    padding = (
        0,
        padded_width - int(original_width),
        0,
        padded_height - int(original_height),
    )
    if any(padding):
        # LTXVLaplacianPyramidBlend pads all three inputs on the bottom and
        # right to powers of two before constructing Kornia's pyramids. This
        # is not merely an implementation detail: it changes the lowest
        # frequency color field that crosses an Outpaint boundary.
        generated = F.pad(generated, padding, mode="reflect")
        source = F.pad(source, padding, mode="reflect")
        mask = F.pad(mask, padding, mode="reflect")

    levels = max(
        1,
        min(
            int(levels),
            int(math.log2(max(2, min(generated.shape[-2:])))),
        ),
    )

    def gaussian_blur(tensor):
        channels = int(tensor.shape[1])
        kernel_1d = torch.tensor(
            (1.0, 4.0, 6.0, 4.0, 1.0),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        kernel_2d = (
            kernel_1d[:, None] * kernel_1d[None, :]
        ).div(256.0)
        kernel = kernel_2d.view(1, 1, 5, 5).expand(
            channels,
            1,
            5,
            5,
        )
        pad_mode = (
            "reflect"
            if min(tensor.shape[-2:]) > 2
            else "replicate"
        )
        blurred = F.conv2d(
            F.pad(tensor, (2, 2, 2, 2), mode=pad_mode),
            kernel,
            groups=channels,
        )
        return blurred

    def gaussian_downsample(tensor):
        """Match Kornia pyrdown: blur, then bilinear resize by one half."""
        height, width = tensor.shape[-2:]
        return F.interpolate(
            gaussian_blur(tensor),
            size=(int(height) // 2, int(width) // 2),
            mode="bilinear",
            align_corners=False,
        )

    def gaussian_upsample(tensor):
        """Match Kornia PyrUp: bilinear resize by two, then blur."""
        height, width = tensor.shape[-2:]
        return gaussian_blur(
            F.interpolate(
                tensor,
                size=(int(height) * 2, int(width) * 2),
                mode="bilinear",
                align_corners=False,
            )
        )

    def gaussian_pyramid(tensor):
        pyramid = [tensor]
        for _ in range(1, levels):
            if min(pyramid[-1].shape[-2:]) <= 8:
                break
            pyramid.append(gaussian_downsample(pyramid[-1]))
        return pyramid

    def laplacian_pyramid(tensor):
        gaussian = gaussian_pyramid(tensor)
        pyramid = [
            current
            - gaussian_upsample(following)
            for current, following in zip(gaussian[:-1], gaussian[1:])
        ]
        pyramid.append(gaussian[-1])
        return pyramid

    generated_pyramid = laplacian_pyramid(generated)
    source_pyramid = laplacian_pyramid(source)
    mask_pyramid = gaussian_pyramid(mask)
    blended = [
        generated_level * mask_level
        + source_level * (1.0 - mask_level)
        for generated_level, source_level, mask_level in zip(
            generated_pyramid,
            source_pyramid,
            mask_pyramid,
        )
    ]
    result = blended[-1]
    for level in reversed(blended[:-1]):
        result = gaussian_upsample(result) + level
    return (
        result[..., :original_height, :original_width]
        .clamp(0.0, 1.0)
        .permute(1, 0, 2, 3)
    )


def _median_float(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) * 0.5


def _frame_opponent_chroma_medians(
    sample: torch.Tensor,
) -> list[tuple[float, float] | None]:
    """Return robust R-Y/B-Y medians for each sampled frame."""
    luma = (
        sample[0] * 0.2126
        + sample[1] * 0.7152
        + sample[2] * 0.0722
    )
    values: list[tuple[float, float] | None] = []
    for frame_no in range(int(sample.shape[1])):
        frame_luma = luma[frame_no]
        valid = (frame_luma >= 0.06) & (frame_luma <= 0.94)
        if int(valid.sum()) < 32:
            values.append(None)
            continue
        values.append(
            (
                float((sample[0, frame_no] - frame_luma)[valid].median()),
                float((sample[2, frame_no] - frame_luma)[valid].median()),
            )
        )
    return values


def _robust_opponent_chroma_stats(
    sample: torch.Tensor,
) -> tuple[float, float, float, float] | None:
    """Return robust R-Y/B-Y centers and interquartile ranges."""
    if sample.shape[0] < 3 or sample.numel() == 0:
        return None
    luma = (
        sample[0] * 0.2126
        + sample[1] * 0.7152
        + sample[2] * 0.0722
    )
    valid = (luma >= 0.04) & (luma <= 0.95)
    if int(valid.sum()) < 256:
        return None
    red = (sample[0] - luma)[valid].float()
    blue = (sample[2] - luma)[valid].float()
    red_quartiles = torch.quantile(
        red,
        torch.tensor((0.25, 0.50, 0.75), device=red.device),
    )
    blue_quartiles = torch.quantile(
        blue,
        torch.tensor((0.25, 0.50, 0.75), device=blue.device),
    )
    return (
        float(red_quartiles[1]),
        float((red_quartiles[2] - red_quartiles[0]).clamp_min(1e-4)),
        float(blue_quartiles[1]),
        float((blue_quartiles[2] - blue_quartiles[0]).clamp_min(1e-4)),
    )


def _estimate_side_chroma_transfers(
    generated: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
    gain: float,
    offset: float,
    sample_frames: int = 16,
    sample_long_side: int = 192,
) -> tuple[
    tuple[float, float, float, float] | None,
    tuple[float, float, float, float] | None,
    tuple[float, float, float, float] | None,
    tuple[float, float, float, float] | None,
]:
    """Match each generated side's chroma distribution to nearby source.

    Marker contamination changes both the center and spread of color in a
    large Outpaint band. Additive blue restoration cannot fix that compressed,
    over-saturated distribution. A robust affine match in opponent-chroma
    space restores the source video's color grade while leaving luminance and
    protected pixels untouched.
    """
    empty = (None, None, None, None)
    binary = mask[:1, :1].detach().float().cpu() >= 0.5
    protected_points = torch.nonzero(~binary[0, 0], as_tuple=False)
    if protected_points.numel() == 0:
        return empty
    top = int(protected_points[:, 0].min())
    bottom = int(protected_points[:, 0].max()) + 1
    left = int(protected_points[:, 1].min())
    right = int(protected_points[:, 1].max()) + 1
    height = min(int(generated.shape[-2]), int(source.shape[-2]))
    width = min(int(generated.shape[-1]), int(source.shape[-1]))
    frames = min(
        int(generated.shape[1]),
        int(source.shape[1]),
        int(mask.shape[1]),
    )
    if frames <= 0:
        return empty
    frame_count = min(max(1, int(sample_frames)), frames)
    frame_indices = (
        torch.linspace(
            0,
            frames - 1,
            frame_count,
            device=binary.device,
        )
        .round()
        .to(dtype=torch.long)
        .unique()
    )
    generated_cpu = generated.detach().cpu()
    source_cpu = source.detach().cpu()

    def sample_region(
        video: torch.Tensor,
        y_slice: slice,
        x_slice: slice,
        *,
        match_exposure: bool,
    ) -> torch.Tensor | None:
        region_height = y_slice.stop - y_slice.start
        region_width = x_slice.stop - x_slice.start
        if region_height < 8 or region_width < 8:
            return None
        spatial_stride = max(
            1,
            int(
                math.ceil(
                    max(region_height, region_width)
                    / max(8, int(sample_long_side))
                )
            ),
        )
        sample = _to_unit_video(
            video[
                :3,
                frame_indices,
                y_slice.start:y_slice.stop:spatial_stride,
                x_slice.start:x_slice.stop:spatial_stride,
            ]
        )
        if match_exposure:
            sample = sample.mul(float(gain)).add(float(offset)).clamp(0.0, 1.0)
        return sample

    middle_y = top + max(1, (bottom - top) // 2)
    middle_x = left + max(1, (right - left) // 2)
    regions = (
        (
            slice(0, top),
            slice(left, right),
            slice(top, middle_y),
            slice(left, right),
        ),
        (
            slice(bottom, height),
            slice(left, right),
            slice(middle_y, bottom),
            slice(left, right),
        ),
        (
            slice(top, bottom),
            slice(0, left),
            slice(top, bottom),
            slice(left, middle_x),
        ),
        (
            slice(top, bottom),
            slice(right, width),
            slice(top, bottom),
            slice(middle_x, right),
        ),
    )
    transfers: list[tuple[float, float, float, float] | None] = []
    for generated_y, generated_x, source_y, source_x in regions:
        generated_sample = sample_region(
            generated_cpu,
            generated_y,
            generated_x,
            match_exposure=True,
        )
        source_sample = sample_region(
            source_cpu,
            source_y,
            source_x,
            match_exposure=False,
        )
        if generated_sample is None or source_sample is None:
            transfers.append(None)
            continue
        generated_stats = _robust_opponent_chroma_stats(generated_sample)
        source_stats = _robust_opponent_chroma_stats(source_sample)
        if generated_stats is None or source_stats is None:
            transfers.append(None)
            continue
        (
            generated_red_center,
            generated_red_iqr,
            generated_blue_center,
            generated_blue_iqr,
        ) = generated_stats
        (
            source_red_center,
            source_red_iqr,
            source_blue_center,
            source_blue_iqr,
        ) = source_stats
        blue_deficit = source_blue_center - generated_blue_center
        # This correction targets the official green/yellow marker signature:
        # generated B-Y is substantially below the nearby source grade. Do not
        # flatten deliberately colorful novel regions without that signature.
        if (
            blue_deficit < 0.025
            or generated_blue_iqr < source_blue_iqr * 1.50
        ):
            transfers.append(None)
            continue
        red_scale = 1.0
        match_red_distribution = (
            generated_red_iqr >= source_red_iqr * 1.35
        )
        if match_red_distribution:
            red_scale = max(
                0.45,
                min(1.0, source_red_iqr / max(generated_red_iqr, 1e-4)),
            )
        blue_scale = max(
            0.30,
            min(1.0, source_blue_iqr / max(generated_blue_iqr, 1e-4)),
        )
        strength = 0.90
        red_gain = 1.0 + (red_scale - 1.0) * strength
        blue_gain = 1.0 + (blue_scale - 1.0) * strength
        red_offset = 0.0
        if match_red_distribution:
            red_offset = (
                source_red_center - red_scale * generated_red_center
            ) * strength
        blue_offset = (
            source_blue_center - blue_scale * generated_blue_center
        ) * strength
        transfers.append(
            (
                red_gain,
                max(-0.18, min(0.18, red_offset)),
                blue_gain,
                max(-0.18, min(0.18, blue_offset)),
            )
        )
    return tuple(transfers)


def _estimate_boundary_chroma_shift(
    generated: torch.Tensor,
    source: torch.Tensor,
    frame_indices: torch.Tensor,
    protected_rect: tuple[int, int, int, int],
    gain: float,
    offset: float,
    seam_width: int = 24,
    sample_long_side: int = 192,
) -> tuple[float, float, float, int] | None:
    """Estimate marker spill from strips immediately across source seams.

    The generated-only canvas can retain the official green missing-region
    marker even when LTX renders the protected rectangle correctly. In that
    case a protected-pixel comparison sees no cast. Adjacent strips on each
    side of the source boundary expose the discontinuity directly. Medians
    across pixels, frames, and available sides keep people crossing the seam
    from controlling the clip-wide correction.
    """
    top, bottom, left, right = protected_rect
    height = min(int(generated.shape[-2]), int(source.shape[-2]))
    width = min(int(generated.shape[-1]), int(source.shape[-1]))
    protected_height = max(0, bottom - top)
    protected_width = max(0, right - left)
    requested_width = max(4, int(seam_width))
    sides: list[tuple[slice, slice, slice, slice]] = []

    horizontal_inset = min(
        requested_width,
        max(0, (protected_width - 16) // 4),
    )
    horizontal_start = left + horizontal_inset
    horizontal_end = right - horizontal_inset
    if horizontal_end - horizontal_start >= 8:
        top_width = min(requested_width, top, protected_height)
        if top_width >= 4:
            sides.append(
                (
                    slice(top, top + top_width),
                    slice(horizontal_start, horizontal_end),
                    slice(top - top_width, top),
                    slice(horizontal_start, horizontal_end),
                )
            )
        bottom_width = min(
            requested_width,
            height - bottom,
            protected_height,
        )
        if bottom_width >= 4:
            sides.append(
                (
                    slice(bottom - bottom_width, bottom),
                    slice(horizontal_start, horizontal_end),
                    slice(bottom, bottom + bottom_width),
                    slice(horizontal_start, horizontal_end),
                )
            )

    vertical_inset = min(
        requested_width,
        max(0, (protected_height - 16) // 4),
    )
    vertical_start = top + vertical_inset
    vertical_end = bottom - vertical_inset
    if vertical_end - vertical_start >= 8:
        left_width = min(requested_width, left, protected_width)
        if left_width >= 4:
            sides.append(
                (
                    slice(vertical_start, vertical_end),
                    slice(left, left + left_width),
                    slice(vertical_start, vertical_end),
                    slice(left - left_width, left),
                )
            )
        right_width = min(
            requested_width,
            width - right,
            protected_width,
        )
        if right_width >= 4:
            sides.append(
                (
                    slice(vertical_start, vertical_end),
                    slice(right - right_width, right),
                    slice(vertical_start, vertical_end),
                    slice(right, right + right_width),
                )
            )

    side_red_shifts: list[float] = []
    side_blue_shifts: list[float] = []
    side_stabilities: list[float] = []
    generated_cpu = generated.detach().cpu()
    source_cpu = source.detach().cpu()
    for source_y, source_x, generated_y, generated_x in sides:
        region_height = max(
            source_y.stop - source_y.start,
            generated_y.stop - generated_y.start,
        )
        region_width = max(
            source_x.stop - source_x.start,
            generated_x.stop - generated_x.start,
        )
        spatial_stride = max(
            1,
            int(
                math.ceil(
                    max(region_height, region_width)
                    / max(8, int(sample_long_side))
                )
            ),
        )
        source_sample = _to_unit_video(
            source_cpu[
                :3,
                frame_indices,
                source_y.start:source_y.stop:spatial_stride,
                source_x.start:source_x.stop:spatial_stride,
            ]
        )
        generated_sample = _to_unit_video(
            generated_cpu[
                :3,
                frame_indices,
                generated_y.start:generated_y.stop:spatial_stride,
                generated_x.start:generated_x.stop:spatial_stride,
            ]
        ).mul(float(gain)).add(float(offset)).clamp(0.0, 1.0)
        source_chroma = _frame_opponent_chroma_medians(source_sample)
        generated_chroma = _frame_opponent_chroma_medians(generated_sample)
        red_shifts: list[float] = []
        blue_shifts: list[float] = []
        for source_value, generated_value in zip(
            source_chroma,
            generated_chroma,
        ):
            if source_value is None or generated_value is None:
                continue
            red_shifts.append(source_value[0] - generated_value[0])
            blue_shifts.append(source_value[1] - generated_value[1])
        minimum_frames = max(1, min(3, int(frame_indices.numel()) // 4))
        if len(red_shifts) < minimum_frames:
            continue
        side_red = _median_float(red_shifts)
        side_blue = _median_float(blue_shifts)
        red_mad = _median_float(
            [abs(value - side_red) for value in red_shifts]
        )
        blue_mad = _median_float(
            [abs(value - side_blue) for value in blue_shifts]
        )
        stability = max(
            0.25,
            min(1.0, 1.0 - max(red_mad, blue_mad) / 0.08),
        )
        side_red_shifts.append(side_red)
        side_blue_shifts.append(side_blue)
        side_stabilities.append(stability)

    side_count = len(side_red_shifts)
    if side_count == 0:
        return None
    red_shift = _median_float(side_red_shifts)
    blue_shift = _median_float(side_blue_shifts)
    side_spread = max(
        _median_float(
            [abs(value - red_shift) for value in side_red_shifts]
        ),
        _median_float(
            [abs(value - blue_shift) for value in side_blue_shifts]
        ),
    )
    confidence = sum(side_stabilities) / side_count
    confidence *= max(0.35, min(1.0, 1.0 - side_spread / 0.10))
    if side_count == 1:
        confidence *= 0.70
    return red_shift, blue_shift, float(confidence), side_count


def _estimate_canvas_chroma_gradients(
    generated: torch.Tensor,
    frame_indices: torch.Tensor,
    protected_rect: tuple[int, int, int, int],
    gain: float,
    offset: float,
    seam_width: int = 24,
    sample_long_side: int = 192,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Detect green-marker chroma that strengthens away from a seam.

    A uniform seam correction cannot remove sentinel contamination whose
    strength grows through a large generated band. Measure each side
    independently and extrapolate only a consistent, blue-deficient trend.
    The returned values are maximum edge corrections for top, bottom, left,
    and right; undetected sides remain exactly zero.
    """
    top, bottom, left, right = protected_rect
    height = int(generated.shape[-2])
    width = int(generated.shape[-1])
    requested_width = max(4, int(seam_width))
    horizontal_inset = min(
        requested_width,
        max(0, (right - left - 16) // 4),
    )
    vertical_inset = min(
        requested_width,
        max(0, (bottom - top - 16) // 4),
    )
    generated_cpu = generated.detach().cpu()
    max_gradient_shift = max(
        0.0,
        float(
            LTX2_OUTPAINTING_CANVAS_MATCH_MAX_GRADIENT_CHROMA_SHIFT
        ),
    )
    fractions = (0.20, 0.40, 0.60, 0.80)

    # (depth, horizontal, long-start, long-end, seam-coordinate, direction)
    specs = (
        (
            top,
            True,
            left + horizontal_inset,
            right - horizontal_inset,
            top,
            -1,
        ),
        (
            height - bottom,
            True,
            left + horizontal_inset,
            right - horizontal_inset,
            bottom,
            1,
        ),
        (
            left,
            False,
            top + vertical_inset,
            bottom - vertical_inset,
            left,
            -1,
        ),
        (
            width - right,
            False,
            top + vertical_inset,
            bottom - vertical_inset,
            right,
            1,
        ),
    )

    def sample_band(
        horizontal: bool,
        long_start: int,
        long_end: int,
        seam: int,
        direction: int,
        band_width: int,
        distance: int,
    ) -> torch.Tensor:
        if direction < 0:
            band_end = seam - distance
            band_start = band_end - band_width
        else:
            band_start = seam + distance
            band_end = band_start + band_width
        if horizontal:
            y_slice = slice(band_start, band_end)
            x_slice = slice(long_start, long_end)
        else:
            y_slice = slice(long_start, long_end)
            x_slice = slice(band_start, band_end)
        region_height = y_slice.stop - y_slice.start
        region_width = x_slice.stop - x_slice.start
        spatial_stride = max(
            1,
            int(
                math.ceil(
                    max(region_height, region_width)
                    / max(8, int(sample_long_side))
                )
            ),
        )
        return _to_unit_video(
            generated_cpu[
                :3,
                frame_indices,
                y_slice.start:y_slice.stop:spatial_stride,
                x_slice.start:x_slice.stop:spatial_stride,
            ]
        ).mul(float(gain)).add(float(offset)).clamp(0.0, 1.0)

    gradients: list[tuple[float, float]] = []
    for (
        depth,
        horizontal,
        long_start,
        long_end,
        seam,
        direction,
    ) in specs:
        band_width = min(requested_width, int(depth))
        if band_width < 4 or long_end - long_start < 8:
            gradients.append((0.0, 0.0))
            continue
        near_sample = sample_band(
            horizontal,
            long_start,
            long_end,
            seam,
            direction,
            band_width,
            0,
        )
        near_chroma = _frame_opponent_chroma_medians(near_sample)
        blue_edge_candidates: list[float] = []
        supporting_distances = 0
        travel = max(0, int(depth) - band_width)
        for fraction in fractions:
            distance = min(
                travel,
                max(1, int(round(travel * fraction))),
            )
            deep_sample = sample_band(
                horizontal,
                long_start,
                long_end,
                seam,
                direction,
                band_width,
                distance,
            )
            deep_chroma = _frame_opponent_chroma_medians(deep_sample)
            blue_shifts: list[float] = []
            deep_blue_values: list[float] = []
            for near_value, deep_value in zip(
                near_chroma,
                deep_chroma,
            ):
                if near_value is None or deep_value is None:
                    continue
                blue_shifts.append(near_value[1] - deep_value[1])
                deep_blue_values.append(deep_value[1])
            if not blue_shifts:
                continue
            blue_shift = _median_float(blue_shifts)
            deep_blue = _median_float(deep_blue_values)
            # The official green sentinel is strongly blue-deficient. Require
            # both an absolute marker-like chroma and a growing discontinuity
            # so naturally changing skies or colored scenery are left alone.
            if blue_shift < 0.035 or deep_blue > -0.12:
                continue
            supporting_distances += 1
            distance_scale = math.sqrt(max(0.05, float(fraction)))
            blue_edge_candidates.append(blue_shift / distance_scale)

        if supporting_distances < 2 or not blue_edge_candidates:
            gradients.append((0.0, 0.0))
            continue
        blue_edge = min(
            max_gradient_shift,
            # A square-root spatial ramp is intentionally conservative near
            # the seam; a small headroom factor lets the detected far-band
            # correction reach the measured median before the hard cap.
            max(0.0, _median_float(blue_edge_candidates) * 1.15),
        )
        # The visible residual is the marker's severe blue deficiency. Red
        # varies naturally between wood, skin, and floors; extrapolating that
        # weaker axis can turn the correction yellow or magenta.
        gradients.append((0.0, blue_edge))

    return tuple(gradients)


def _estimate_canvas_match(
    generated: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
    source_inset: int,
    sample_frames: int = 16,
    sample_long_side: int = 128,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    int,
    tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
] | None:
    """Measure a stable exposure/detail match and generated-canvas cast.

    Corresponding protected pixels remain the best exposure reference when
    they correlate with the source. Chroma additionally uses adjacent seam
    strips because the green IC-LoRA sentinel can contaminate only the novel
    canvas. A low-correlation sliding window is never allowed to apply a
    full-strength correction in the opposite direction.
    """
    binary = mask[:1, :1].detach().float().cpu() >= 0.5
    protected_points = torch.nonzero(~binary[0, 0], as_tuple=False)
    if protected_points.numel() == 0:
        return None

    protected_top = int(protected_points[:, 0].min())
    protected_bottom = int(protected_points[:, 0].max()) + 1
    protected_left = int(protected_points[:, 1].min())
    protected_right = int(protected_points[:, 1].max()) + 1
    protected_rect = (
        protected_top,
        protected_bottom,
        protected_left,
        protected_right,
    )
    height = protected_bottom - protected_top
    width = protected_right - protected_left
    inset = min(
        max(0, int(source_inset)),
        max(0, (height - 8) // 2),
        max(0, (width - 8) // 2),
    )
    top = protected_top + inset
    bottom = protected_bottom - inset
    left = protected_left + inset
    right = protected_right - inset
    if bottom - top < 8 or right - left < 8:
        return None

    frames = min(
        int(generated.shape[1]),
        int(source.shape[1]),
        int(mask.shape[1]),
    )
    if frames <= 0:
        return None
    frame_count = min(max(1, int(sample_frames)), frames)
    frame_indices = (
        torch.linspace(
            0,
            frames - 1,
            frame_count,
            device=binary.device,
        )
        .round()
        .to(dtype=torch.long)
        .unique()
    )
    spatial_stride = max(
        1,
        int(
            math.ceil(
                max(bottom - top, right - left)
                / max(8, int(sample_long_side))
            )
        ),
    )
    generated_sample = _to_unit_video(
        generated.detach().cpu()[
            :3,
            frame_indices,
            top:bottom:spatial_stride,
            left:right:spatial_stride,
        ]
    )
    source_sample = _to_unit_video(
        source.detach().cpu()[
            :3,
            frame_indices,
            top:bottom:spatial_stride,
            left:right:spatial_stride,
        ]
    )
    luma_weights = torch.tensor(
        (0.2126, 0.7152, 0.0722),
        dtype=torch.float32,
        device=generated_sample.device,
    ).view(3, 1, 1, 1)
    generated_luma = (generated_sample * luma_weights).sum(dim=0)
    source_luma = (source_sample * luma_weights).sum(dim=0)
    generated_mean = generated_luma.mean()
    source_mean = source_luma.mean()
    generated_centered = generated_luma - generated_mean
    source_centered = source_luma - source_mean
    generated_std = generated_centered.square().mean().sqrt()
    source_std = source_centered.square().mean().sqrt()

    correlation = torch.tensor(0.0)
    protected_reliable = bool(
        generated_std >= 0.02 and source_std >= 0.02
    )
    if protected_reliable:
        correlation = (
            (generated_centered * source_centered).mean()
            / (generated_std * source_std).clamp_min(1e-6)
        )
        protected_reliable = bool(
            torch.isfinite(correlation) and float(correlation) >= 0.55
        )

    gain = 1.0
    offset = 0.0
    protected_red_shift = 0.0
    protected_blue_shift = 0.0
    sharpen = 0.0
    if protected_reliable:
        max_gain = max(
            1.0,
            float(LTX2_OUTPAINTING_CANVAS_MATCH_MAX_GAIN),
        )
        gain = float(
            (source_std / generated_std).clamp(1.0 / max_gain, max_gain)
        )
        max_offset = max(
            0.0,
            float(LTX2_OUTPAINTING_CANVAS_MATCH_MAX_OFFSET),
        )
        offset = float(
            (source_mean - generated_mean * gain).clamp(
                -max_offset,
                max_offset,
            )
        )
        matched_sample = (
            generated_sample.mul(gain).add(offset).clamp(0.0, 1.0)
        )
        matched_luma = (matched_sample * luma_weights).sum(dim=0)
        protected_red_shift = float(
            (source_sample[0] - source_luma).mean()
            - (matched_sample[0] - matched_luma).mean()
        )
        protected_blue_shift = float(
            (source_sample[2] - source_luma).mean()
            - (matched_sample[2] - matched_luma).mean()
        )

        def detail_energy(luma: torch.Tensor) -> torch.Tensor:
            horizontal = (
                luma[..., :, 1:] - luma[..., :, :-1]
            ).abs().mean()
            vertical = (
                luma[..., 1:, :] - luma[..., :-1, :]
            ).abs().mean()
            return (horizontal + vertical) * 0.5

        generated_detail = detail_energy(generated_luma)
        source_detail = detail_energy(source_luma)
        max_sharpen = max(
            0.0,
            float(LTX2_OUTPAINTING_CANVAS_MATCH_MAX_SHARPEN),
        )
        if generated_detail > 1e-4 and source_detail > generated_detail:
            sharpen = float(
                (source_detail / generated_detail - 1.0).clamp(
                    0.0,
                    max_sharpen,
                )
            )

    boundary_match = _estimate_boundary_chroma_shift(
        generated,
        source,
        frame_indices,
        protected_rect,
        gain,
        offset,
        seam_width=max(8, int(source_inset)),
    )
    boundary_confidence = 0.0
    boundary_sides = 0
    if boundary_match is None:
        if not protected_reliable:
            return None
        red_chroma_shift = protected_red_shift
        blue_chroma_shift = protected_blue_shift
    else:
        (
            boundary_red_shift,
            boundary_blue_shift,
            boundary_confidence,
            boundary_sides,
        ) = boundary_match
        if protected_reliable:
            base_weight = 0.85 if boundary_sides >= 2 else 0.70
            boundary_weight = base_weight * (
                0.65 + 0.35 * boundary_confidence
            )
            red_chroma_shift = (
                protected_red_shift * (1.0 - boundary_weight)
                + boundary_red_shift * boundary_weight
            )
            blue_chroma_shift = (
                protected_blue_shift * (1.0 - boundary_weight)
                + boundary_blue_shift * boundary_weight
            )
        else:
            # A mismatched/padded sliding window caused the old estimator to
            # hit the chroma clamp in the wrong direction. Seam medians do not
            # require the model's protected rendering to match temporally.
            red_chroma_shift = boundary_red_shift
            blue_chroma_shift = boundary_blue_shift

    max_chroma_shift = max(
        0.0,
        float(LTX2_OUTPAINTING_CANVAS_MATCH_MAX_CHROMA_SHIFT),
    )
    red_chroma_shift = max(
        -max_chroma_shift,
        min(max_chroma_shift, float(red_chroma_shift)),
    )
    blue_chroma_shift = max(
        -max_chroma_shift,
        min(max_chroma_shift, float(blue_chroma_shift)),
    )
    edge_chroma_shifts = _estimate_canvas_chroma_gradients(
        generated,
        frame_indices,
        protected_rect,
        gain,
        offset,
        seam_width=max(8, int(source_inset)),
    )
    return (
        gain,
        offset,
        red_chroma_shift,
        blue_chroma_shift,
        sharpen,
        float(correlation),
        boundary_confidence,
        boundary_sides,
        edge_chroma_shifts,
    )


def _apply_opponent_chroma_shift(
    corrected: torch.Tensor,
    red_chroma_shift,
    blue_chroma_shift,
) -> torch.Tensor:
    """Apply R-Y/B-Y shifts while preserving Rec.709 luminance."""
    red_weight = 0.2126
    green_weight = 0.7152
    blue_weight = 0.0722
    rgb = corrected[:3]
    luma = (
        rgb[0] * red_weight
        + rgb[1] * green_weight
        + rgb[2] * blue_weight
    )
    # Adding chroma to near-black pixels turns neutral rafters into colored
    # blocks. Fade every correction smoothly through the deep shadows.
    chroma_weight = (luma - 0.03).div(0.22).clamp(0.0, 1.0)
    chroma_weight = (
        chroma_weight.square() * (3.0 - 2.0 * chroma_weight)
    )
    red = (
        luma
        + (rgb[0] - luma)
        + red_chroma_shift * chroma_weight
    )
    blue = (
        luma
        + (rgb[2] - luma)
        + blue_chroma_shift * chroma_weight
    )
    green = (
        luma - red_weight * red - blue_weight * blue
    ).div(green_weight)
    shifted = corrected.clone()
    shifted[:3] = torch.stack((red, green, blue), dim=0)
    return shifted


def _apply_side_chroma_transfers(
    corrected: torch.Tensor,
    mask: torch.Tensor,
    transfers: tuple[
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
    ],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Apply robust affine chroma matches to active Outpaint sides."""
    if corrected.shape[0] < 3 or not any(
        transfer is not None for transfer in transfers
    ):
        return None
    binary = mask[:1, :1].detach().float().cpu() >= 0.5
    protected_points = torch.nonzero(~binary[0, 0], as_tuple=False)
    if protected_points.numel() == 0:
        return None
    top = int(protected_points[:, 0].min())
    bottom = int(protected_points[:, 0].max()) + 1
    left = int(protected_points[:, 1].min())
    right = int(protected_points[:, 1].max()) + 1
    height = int(corrected.shape[-2])
    width = int(corrected.shape[-1])

    red_gain_sum = torch.zeros(
        (height, width),
        dtype=corrected.dtype,
        device=corrected.device,
    )
    red_offset_sum = torch.zeros_like(red_gain_sum)
    blue_gain_sum = torch.zeros_like(red_gain_sum)
    blue_offset_sum = torch.zeros_like(red_gain_sum)
    transfer_count = torch.zeros_like(red_gain_sum)

    def add_region(
        y_slice: slice,
        x_slice: slice,
        transfer: tuple[float, float, float, float] | None,
    ) -> None:
        if transfer is None:
            return
        red_gain, red_offset, blue_gain, blue_offset = transfer
        red_gain_sum[y_slice, x_slice].add_(float(red_gain))
        red_offset_sum[y_slice, x_slice].add_(float(red_offset))
        blue_gain_sum[y_slice, x_slice].add_(float(blue_gain))
        blue_offset_sum[y_slice, x_slice].add_(float(blue_offset))
        transfer_count[y_slice, x_slice].add_(1.0)

    top_transfer, bottom_transfer, left_transfer, right_transfer = transfers
    if top > 0:
        add_region(slice(0, top), slice(0, width), top_transfer)
    if bottom < height:
        add_region(
            slice(bottom, height),
            slice(0, width),
            bottom_transfer,
        )
    if left > 0:
        add_region(slice(0, height), slice(0, left), left_transfer)
    if right < width:
        add_region(
            slice(0, height),
            slice(right, width),
            right_transfer,
        )

    active = transfer_count > 0
    if not bool(active.any()):
        return None
    divisor = transfer_count.clamp_min(1.0)
    red_gain = torch.where(
        active,
        red_gain_sum / divisor,
        torch.ones_like(red_gain_sum),
    )
    red_offset = red_offset_sum / divisor
    blue_gain = torch.where(
        active,
        blue_gain_sum / divisor,
        torch.ones_like(blue_gain_sum),
    )
    blue_offset = blue_offset_sum / divisor

    rgb = corrected[:3]
    luma = (
        rgb[0] * 0.2126
        + rgb[1] * 0.7152
        + rgb[2] * 0.0722
    )
    red_chroma = rgb[0] - luma
    blue_chroma = rgb[2] - luma
    canvas = mask[0, : corrected.shape[1]].to(
        device=corrected.device,
        dtype=torch.bool,
    )
    coverage = active.unsqueeze(0) & canvas
    red_shift = (
        red_chroma * red_gain.unsqueeze(0)
        + red_offset.unsqueeze(0)
        - red_chroma
    ) * coverage
    blue_shift = (
        blue_chroma * blue_gain.unsqueeze(0)
        + blue_offset.unsqueeze(0)
        - blue_chroma
    ) * coverage
    return (
        _apply_opponent_chroma_shift(
            corrected,
            red_shift,
            blue_shift,
        ),
        coverage,
    )


def _apply_canvas_match(
    generated_unit: torch.Tensor,
    gain: float,
    offset: float,
    red_chroma_shift: float,
    blue_chroma_shift: float,
    sharpen: float,
    mask: torch.Tensor | None = None,
    edge_chroma_shifts: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] | None = None,
    side_chroma_transfers: tuple[
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
        tuple[float, float, float, float] | None,
    ] | None = None,
) -> torch.Tensor:
    """Apply a bounded clip-wide exposure, chroma, and detail correction."""
    exposure_matched = generated_unit.mul(float(gain)).add(float(offset))
    corrected = exposure_matched
    if (
        corrected.shape[0] >= 3
        and (
            abs(float(red_chroma_shift)) > 1e-5
            or abs(float(blue_chroma_shift)) > 1e-5
        )
    ):
        corrected = _apply_opponent_chroma_shift(
            corrected,
            float(red_chroma_shift),
            float(blue_chroma_shift),
        )
    if (
        corrected.shape[0] >= 3
        and mask is not None
        and edge_chroma_shifts is not None
        and any(
            abs(float(red_shift)) > 1e-5
            or abs(float(blue_shift)) > 1e-5
            for red_shift, blue_shift in edge_chroma_shifts
        )
    ):
        binary = mask[:1, :1].detach().float().cpu() >= 0.5
        protected_points = torch.nonzero(~binary[0, 0], as_tuple=False)
        if protected_points.numel() > 0:
            top = int(protected_points[:, 0].min())
            bottom = int(protected_points[:, 0].max()) + 1
            left = int(protected_points[:, 1].min())
            right = int(protected_points[:, 1].max()) + 1
            height = int(corrected.shape[-2])
            width = int(corrected.shape[-1])
            red_map = torch.zeros(
                (height, width),
                dtype=corrected.dtype,
                device=corrected.device,
            )
            blue_map = torch.zeros_like(red_map)

            def merge_region(
                y_slice: slice,
                x_slice: slice,
                ramp: torch.Tensor,
                red_shift: float,
                blue_shift: float,
            ) -> None:
                if abs(float(red_shift)) > 1e-5:
                    red_map[y_slice, x_slice] = torch.maximum(
                        red_map[y_slice, x_slice],
                        ramp * float(red_shift),
                    )
                if abs(float(blue_shift)) > 1e-5:
                    blue_map[y_slice, x_slice] = torch.maximum(
                        blue_map[y_slice, x_slice],
                        ramp * float(blue_shift),
                    )

            top_shift, bottom_shift, left_shift, right_shift = (
                edge_chroma_shifts
            )
            if top > 0:
                ramp = torch.linspace(
                    1.0,
                    1.0 / max(1.0, float(top)),
                    top,
                    dtype=corrected.dtype,
                    device=corrected.device,
                ).sqrt().view(top, 1)
                merge_region(
                    slice(0, top),
                    slice(0, width),
                    ramp,
                    top_shift[0],
                    top_shift[1],
                )
            if bottom < height:
                depth = height - bottom
                ramp = torch.linspace(
                    1.0 / max(1.0, float(depth)),
                    1.0,
                    depth,
                    dtype=corrected.dtype,
                    device=corrected.device,
                ).sqrt().view(depth, 1)
                merge_region(
                    slice(bottom, height),
                    slice(0, width),
                    ramp,
                    bottom_shift[0],
                    bottom_shift[1],
                )
            if left > 0:
                ramp = torch.linspace(
                    1.0,
                    1.0 / max(1.0, float(left)),
                    left,
                    dtype=corrected.dtype,
                    device=corrected.device,
                ).sqrt().view(1, left)
                merge_region(
                    slice(0, height),
                    slice(0, left),
                    ramp,
                    left_shift[0],
                    left_shift[1],
                )
            if right < width:
                depth = width - right
                ramp = torch.linspace(
                    1.0 / max(1.0, float(depth)),
                    1.0,
                    depth,
                    dtype=corrected.dtype,
                    device=corrected.device,
                ).sqrt().view(1, depth)
                merge_region(
                    slice(0, height),
                    slice(right, width),
                    ramp,
                    right_shift[0],
                    right_shift[1],
                )
            canvas = (
                mask[0, : corrected.shape[1]]
                .to(device=corrected.device, dtype=corrected.dtype)
            )
            corrected = _apply_opponent_chroma_shift(
                corrected,
                red_map.unsqueeze(0) * canvas,
                blue_map.unsqueeze(0) * canvas,
            )
    if mask is not None and side_chroma_transfers is not None:
        transferred = _apply_side_chroma_transfers(
            exposure_matched,
            mask,
            side_chroma_transfers,
        )
        if transferred is not None:
            transferred_video, transfer_coverage = transferred
            # The robust affine transfer supersedes additive marker cleanup
            # only on sides where the marker signature was actually found.
            corrected = torch.where(
                transfer_coverage.unsqueeze(0),
                transferred_video,
                corrected,
            )
    if sharpen > 1e-4:
        frames_nchw = corrected.permute(1, 0, 2, 3)
        kernel_1d = torch.tensor(
            (1.0, 2.0, 1.0),
            dtype=frames_nchw.dtype,
            device=frames_nchw.device,
        )
        kernel_2d = (
            kernel_1d[:, None] * kernel_1d[None, :]
        ).div(16.0)
        kernel = kernel_2d.view(1, 1, 3, 3).expand(
            frames_nchw.shape[1],
            1,
            3,
            3,
        )
        padded = F.pad(
            frames_nchw,
            (1, 1, 1, 1),
            mode="replicate",
        )
        blurred = F.conv2d(
            padded,
            kernel,
            groups=frames_nchw.shape[1],
        )
        frames_nchw = frames_nchw.add(
            (frames_nchw - blurred) * float(sharpen)
        )
        corrected = frames_nchw.permute(1, 0, 2, 3)
    return corrected.clamp(0.0, 1.0)


def _apply_ltx2_mask_blend(
    video_tensor: torch.Tensor,
    source: torch.Tensor | None,
    mask: torch.Tensor | None,
    output_frame_num: int,
    height: int,
    width: int,
    mask_low_res_dilation: int = 0,
    source_feather_pixels: int = 24,
    match_generated_canvas: bool = True,
    blend_mode: str = "laplacian",
    full_frame_laplacian: bool = False,
    correct_marker_residue: bool = False,
) -> torch.Tensor:
    """Restore protected source pixels with a bounded multiscale boundary.

    The official implementation blends the complete clip at once.  That
    can temporarily require many gigabytes for a portrait video, even
    though every pyramid operation is frame-independent.  Processing a
    bounded number of frames at a time keeps a predictable RAM ceiling.
    The pass-one handoff may use the complete multiscale result, matching
    Lightricks' reference graph. Final composition can confine that result to
    a narrow source-side seam so people and details deeper in the protected
    source remain unchanged.
    Its optional marker cleanup is deliberately narrower than canvas color
    matching: it only restores a detected blue-chroma deficit that becomes
    stronger away from an Outpaint seam, the signature of the #66FF00 guide.
    """
    if source is None or mask is None:
        return video_tensor
    frames = min(
        int(output_frame_num),
        int(video_tensor.shape[1]),
        int(source.shape[1]),
        int(mask.shape[1]),
    )
    if frames <= 0:
        return video_tensor

    blend_mode = str(blend_mode or "laplacian").strip().lower()
    if blend_mode not in ("laplacian", "gaussian"):
        raise ValueError(f"Unsupported LTX-2 mask blend mode: {blend_mode}")

    source = source.detach().cpu()[:, :frames, :height, :width]
    mask = mask.detach().cpu()[:1, :frames, :height, :width]
    generated = video_tensor[:, :frames, :height, :width]
    result = video_tensor.detach().cpu()
    if torch.is_inference(result):
        result = result.clone()

    canvas_match = None
    side_chroma_transfers = None
    if (
        blend_mode == "laplacian"
        and match_generated_canvas
        and LTX2_OUTPAINTING_CANVAS_MATCH
    ):
        canvas_match = _estimate_canvas_match(
            generated,
            source,
            mask,
            source_inset=max(8, int(source_feather_pixels)),
        )
        if canvas_match is not None:
            (
                gain,
                offset,
                red_chroma_shift,
                blue_chroma_shift,
                sharpen,
                correlation,
                boundary_confidence,
                boundary_sides,
                edge_chroma_shifts,
            ) = canvas_match
            edge_labels = ("top", "bottom", "left", "right")
            edge_summary = ", ".join(
                f"{label} R-Y {red_shift:+.3f}/B-Y {blue_shift:+.3f}"
                for label, (red_shift, blue_shift) in zip(
                    edge_labels,
                    edge_chroma_shifts,
                )
                if abs(float(red_shift)) > 1e-5
                or abs(float(blue_shift)) > 1e-5
            )
            print(
                "[LTX2] Matched generated Outpaint canvas to source "
                f"(gain={gain:.3f}, offset={offset:+.3f}, "
                "chroma="
                f"(R-Y {red_chroma_shift:+.3f}, "
                f"B-Y {blue_chroma_shift:+.3f}), "
                f"detail={sharpen:.3f}, confidence={correlation:.2f})."
                f" Seam calibration used {boundary_sides} side(s) "
                f"at confidence {boundary_confidence:.2f}."
            )
            if edge_summary:
                print(
                    "[LTX2] Correcting detected green-marker gradient: "
                    f"{edge_summary}."
                )
            side_chroma_transfers = _estimate_side_chroma_transfers(
                generated,
                source,
                mask,
                gain,
                offset,
            )
            transfer_summary = ", ".join(
                (
                    f"{label} R-Y gain {transfer[0]:.2f}/offset "
                    f"{transfer[1]:+.3f}, B-Y gain "
                    f"{transfer[2]:.2f}/offset {transfer[3]:+.3f}"
                )
                for label, transfer in zip(
                    edge_labels,
                    side_chroma_transfers,
                )
                if transfer is not None
            )
            if transfer_summary:
                print(
                    "[LTX2] Matching generated Outpaint color distribution: "
                    f"{transfer_summary}."
                )

    marker_edge_chroma_shifts = None
    if correct_marker_residue and canvas_match is None:
        binary = mask[:1, :1].detach().float().cpu() >= 0.5
        protected_points = torch.nonzero(
            ~binary[0, 0],
            as_tuple=False,
        )
        if protected_points.numel() > 0:
            protected_rect = (
                int(protected_points[:, 0].min()),
                int(protected_points[:, 0].max()) + 1,
                int(protected_points[:, 1].min()),
                int(protected_points[:, 1].max()) + 1,
            )
            frame_count = min(16, frames)
            frame_indices = (
                torch.linspace(
                    0,
                    frames - 1,
                    frame_count,
                    device=binary.device,
                )
                .round()
                .to(dtype=torch.long)
                .unique()
            )
            detected_shifts = _estimate_canvas_chroma_gradients(
                generated,
                frame_indices,
                protected_rect,
                gain=1.0,
                offset=0.0,
                seam_width=max(8, int(source_feather_pixels) * 3),
            )
            if any(
                abs(float(red_shift)) > 1e-5
                or abs(float(blue_shift)) > 1e-5
                for red_shift, blue_shift in detected_shifts
            ):
                marker_edge_chroma_shifts = detected_shifts
                side_names = ("top", "bottom", "left", "right")
                summary = ", ".join(
                    f"{name} B-Y {blue_shift:+.3f}"
                    for name, (_, blue_shift) in zip(
                        side_names,
                        detected_shifts,
                    )
                    if abs(float(blue_shift)) > 1e-5
                )
                print(
                    "[LTX2] Removing detected #66FF00 marker residue from "
                    f"generated pixels only: {summary}."
                )

    pixels_per_frame = max(1, int(height) * int(width))
    chunk_frames = max(1, min(8, 8_000_000 // pixels_per_frame))
    for start in range(0, frames, chunk_frames):
        end = min(frames, start + chunk_frames)
        source_unit = _to_unit_video(source[:, start:end])
        generated_unit = _to_unit_video(
            generated[:, start:end].detach().cpu()
        )
        mask_chunk = mask[:, start:end]
        if canvas_match is not None:
            generated_unit = _apply_canvas_match(
                generated_unit,
                canvas_match[0],
                canvas_match[1],
                canvas_match[2],
                canvas_match[3],
                canvas_match[4],
                mask=mask_chunk,
                edge_chroma_shifts=canvas_match[8],
                side_chroma_transfers=side_chroma_transfers,
            )
        if blend_mode == "gaussian":
            feather_alpha = _build_gaussian_margin_alpha(
                mask_chunk,
                source_feather_pixels,
            )
            blended = torch.where(
                mask_chunk > 0,
                generated_unit,
                generated_unit * feather_alpha
                + source_unit * (1.0 - feather_alpha),
            )
        else:
            # The conditioning reference contains a conspicuous missing-area
            # marker. It is irrelevant wherever the mask selects generated
            # pixels, but an ordinary Laplacian pyramid can still carry its
            # low frequencies across the soft boundary. Replace that masked
            # portion with the generated canvas before building the source
            # pyramid. Protected pixels remain untouched, while no sentinel
            # color is available to leak into either side of the seam.
            # Never let the conspicuous missing-area marker enter the source
            # pyramid. This is required for both the full-frame pass-one
            # handoff and the bounded final restore: even when the generated
            # canvas is selected again below, low-frequency marker color can
            # otherwise bleed inward across the source-side seam.
            mask_alpha = mask_chunk.float().clamp(0.0, 1.0)
            pyramid_source = (
                source_unit * (1.0 - mask_alpha)
                + generated_unit * mask_alpha
            )
            blended = _laplacian_pyramid_blend(
                generated_unit.mul(2.0).sub(1.0),
                pyramid_source.mul(2.0).sub(1.0),
                mask_chunk,
                mask_low_res_dilation=mask_low_res_dilation,
            )
            if not full_frame_laplacian:
                feather_alpha = _build_source_boundary_feather_alpha(
                    mask_chunk,
                    source_feather_pixels,
                )
                # Preserve the model result throughout the generated canvas.
                # Inside the protected rectangle, use the multiscale result
                # only within the bounded feather and restore the source
                # beyond it.
                blended = torch.where(
                    mask_chunk > 0,
                    generated_unit,
                    blended * feather_alpha
                    + source_unit * (1.0 - feather_alpha),
                )
        if marker_edge_chroma_shifts is not None:
            # Run after source restoration. The correction map is multiplied
            # by the generation mask, making protected pixels bit-identical
            # to the normal official blend while cleaning only final canvas
            # pixels. Applying it before the Laplacian pyramid could let its
            # low-frequency component cross back into the source rectangle.
            blended = _apply_canvas_match(
                blended,
                gain=1.0,
                offset=0.0,
                red_chroma_shift=0.0,
                blue_chroma_shift=0.0,
                sharpen=0.0,
                mask=mask_chunk,
                edge_chroma_shifts=marker_edge_chroma_shifts,
                side_chroma_transfers=None,
            )
        if result.dtype == torch.uint8:
            blended = (
                blended.mul(255.0)
                .round()
                .clamp(0.0, 255.0)
                .to(dtype=torch.uint8)
            )
        else:
            blended = blended.mul(2.0).sub(1.0).to(dtype=result.dtype)
        result[:, start:end, :height, :width] = blended
    return result
