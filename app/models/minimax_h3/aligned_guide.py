"""Pixel-space preparation for MiniMax H3's frame-aligned guide condition."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prepare_control_video_tensor(video, height: int, width: int) -> torch.Tensor | None:
    """Normalize control video pixels to CPU CTHW in ``[-1, 1]``."""

    if video is None or not isinstance(video, torch.Tensor):
        return None
    source = video
    if source.ndim == 3:
        source = source.unsqueeze(1)
    if source.ndim != 4 or source.shape[1] < 1:
        return None
    source = source.detach().to(device="cpu")
    if source.dtype == torch.uint8:
        source = source.float().div(127.5).sub(1.0)
    else:
        source = source.float()
        if float(source.amin()) >= -0.01 and float(source.amax()) <= 1.01:
            source = source.mul(2.0).sub(1.0)
        source = source.clamp(-1.0, 1.0)
    if tuple(source.shape[-2:]) != (int(height), int(width)):
        source = F.interpolate(
            source.permute(1, 0, 2, 3),
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)
    return source.contiguous()


def aligned_guide_visual_condition(
    input_frames,
    history_count: int,
    target_frame_num: int,
    height: int,
    width: int,
) -> dict:
    """Slice exact source frames after carried history; never pad or shift."""

    source = prepare_control_video_tensor(input_frames, height, width)
    if source is None:
        raise ValueError("BFS Aligned Swap requires readable source-video frames.")
    start = int(history_count)
    count = int(target_frame_num)
    if start < 0 or count < 1:
        raise ValueError("BFS Aligned Swap received invalid source-window geometry.")
    stop = start + count
    if int(source.shape[1]) < stop:
        raise ValueError(
            "BFS Aligned Swap source window is shorter than the requested target; "
            "the aligned guide cannot pad or shift source frames."
        )
    aligned = source[:, start:stop].contiguous()
    if int(aligned.shape[1]) != count:
        raise ValueError("BFS Aligned Swap could not map every target frame to its source frame.")
    return {"anchor": "first", "source": aligned, "keep_all_latents": True}
