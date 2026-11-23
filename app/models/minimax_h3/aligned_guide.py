"""Pixel-space preparation for MiniMax H3's frame-aligned guide condition."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .packing import MINIMAX_H3_FRAMES_PER_CHUNK, align_num_frames


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
    """Slice exact source frames and fill only H3's internal alignment tail."""

    source = prepare_control_video_tensor(input_frames, height, width)
    if source is None:
        raise ValueError("BFS Aligned Swap requires readable source-video frames.")
    start = int(history_count)
    count = int(target_frame_num)
    if start < 0 or count < 1:
        raise ValueError("BFS Aligned Swap received invalid source-window geometry.")
    stop = start + count
    available = max(0, int(source.shape[1]) - start)
    if available < 1:
        raise ValueError(
            "BFS Aligned Swap source window has no frames at the requested start."
        )
    source_window = source[:, start:min(stop, int(source.shape[1]))]
    missing = count - int(source_window.shape[1])
    if missing > 0:
        # WGP aligns the internal H3 denoise grid upward to 17*n+5 while it
        # keeps the published source duration exact. Pad only that grid tail;
        # all real source frames stay at their original indices.
        if (
            missing > MINIMAX_H3_FRAMES_PER_CHUNK - 1
            or align_num_frames(available) != count
        ):
            raise ValueError(
                "BFS Aligned Swap source window is materially shorter than the "
                "requested H3 target grid."
            )
        source_window = torch.cat(
            [
                source_window,
                source_window[:, -1:].expand(
                    -1, missing, -1, -1
                ),
            ],
            dim=1,
        )
    aligned = source_window.contiguous()
    if int(aligned.shape[1]) != count:
        raise ValueError("BFS Aligned Swap could not map every target frame to its source frame.")
    return {"anchor": "first", "source": aligned, "keep_all_latents": True}
