# Copyright 2024-2026 The Alibaba Wan Team Authors. All rights reserved.
# SCAIL-2 helpers for WanGP.

import concurrent.futures
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from shared.utils.utils import calculate_new_dimensions, convert_image_to_tensor, convert_tensor_to_image, expand_or_shrink_mask, to_rgb_tensor
from ..modules.posemb_layers import get_nd_rotary_pos_embed


SCAIL2_TYPES = {"scail2_14B", "scail2_1.3B"}
SCAIL2_ANIMATE_PREPROCESSING_RAW = "raw"
SCAIL2_ANIMATE_PREPROCESSING_POSE = "pose"
SCAIL2_DEBUG_REF_MASK_PATH = "scail2_image_ref_mask_debug.png"
SCAIL2_DEBUG_MATTED_REF_PATH = "scail2_image_ref_condition_debug.png"
SCAIL2_INFOS = """
# SCAIL-2 Workflows

SCAIL-2 replaces or animates people from a reference image while following a control video and a colored person mask.

## Animate Mode

Use Animate when you want a character to move like the person in the Control Video.

Required inputs:

- Location: use `Start Image` or `Video to Continue`.
- Control Video: the motion or scene you want to follow.
- Video Mask: a colored mask video showing the people to animate.
- Reference Image: optional. If you do not provide one, WanGP builds the reference from the Start Image or the current Continue Video frame.

Animate preprocessing:

- `Use Raw Control Video Content`: default. The model sees the resized Control Video itself. This preserves more scene detail and motion context, but the original people or clothing in the control video can influence the result more.
- `Extract 3D Pose information`: WanGP extracts body pose from the Control Video first. This reduces appearance leakage from the control video, but it can lose details such as props, hands, loose clothing, or subtle scene motion.

## Replacement Mode

Use Replacement when you want to keep the Control Video as the source scene and replace masked people with the reference character.

Required inputs:

- Location: Text Prompt.
- Reference Image: required.
- Control Video: required.
- Persons Locations: Masked Area.
- Video Mask: required.

Replacement uses the Control Video as the source frames. The mask says which people are replaced.

## Experimental Subwindows

SCAIL-2 can use experimental subwindow sampling inside each sliding-window generation. When `Sub Parallel Window Size` is nonzero, WanGP denoises several overlapping temporal subwindows sequentially at every denoising step, blends their noise predictions over `Sub Parallel Window Overlap`, then applies one scheduler step to the full latent window. This can reduce peak VRAM for long SCAIL-2 windows, but it may introduce motion or identity discontinuities at subwindow boundaries. Keep it disabled unless you are testing longer windows or need the VRAM reduction.

## Colored Masks

SCAIL-2 relies on mask colors. Each person should keep one stable color for the whole video. For multiple people, use one color per person. Do not use a single gray/white mask when you need to distinguish several characters.

The default person colors are blue, red, green, magenta, cyan, and yellow. Background should stay uncolored. In Replacement mode, WanGP converts the background to the model's expected replacement background color.

## Building Masks With Magic Mask

Use the Magic Mask tool on the Video Mask input:

- Enter a keyword such as `person`, `woman`, `man`, or a clothing description.
- Generate the mask, then check the preview.
- For multiple people, make sure each person gets a separate color and keeps it across frames.
- If the wrong object is selected, adjust the keyword or edit the mask before generating.

## Troubleshooting

### Image Reference Mask Extraction Fails

If WanGP shows `SCAIL-2 could not extract the image reference mask. Check Image Ref Keyword content.`, fill `Image Ref Keyword content` with a keyword that describes what is visible in the reference frame, usually `person`, `woman`, or `man`. SCAIL-2 uses this keyword to guide SAM 3 reference mask extraction.

## Tips

- Review the Image References preview before generation. SCAIL-2 shows the prepared reference image and the reference mask there.
- For best results, use a Reference Image or a Start Image that is closely aligned to the first frame of the control video. You can use an Image Model generator for this.
- Extra Reference Images are experimental. The first image stays the primary reference; later images are fit into the output canvas and auto-masked with the same Image Ref Keyword content.
- If Animate looks too close to the original control video person, try `Extract 3D Pose information`.
- If pose mode loses important visual details, switch back to `Use Raw Control Video Content`.
- Keep the number of selected people in `Type of Process` aligned with the number of colored people in the mask.
- Use clean masks with stable colors. Flickering colors or missing mask frames usually produce unstable identity or motion.
"""


def test_scail2(base_model_type: str) -> bool:
    return base_model_type in SCAIL2_TYPES


def test_scail2_replace(video_prompt_type: str) -> bool:
    return "0" in (video_prompt_type or "")


def prepare_scail2_mask(mask, frame_count, height, width, device, dtype):
    if mask is None:
        return torch.ones(3, frame_count, height, width, device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1) if mask.shape[0] in (1, 3) and frame_count == 1 else mask.unsqueeze(0)
    elif mask.ndim == 5:
        mask = mask[0]
    if mask.shape[0] == 1:
        mask = mask.expand(3, -1, -1, -1)
    elif mask.shape[0] != 3:
        mask = mask[:1].expand(3, -1, -1, -1)
    if mask.shape[1] != frame_count:
        if mask.shape[1] == 1:
            mask = mask.expand(-1, frame_count, -1, -1)
        elif mask.shape[1] < frame_count:
            mask = torch.cat([mask, mask[:, -1:].expand(-1, frame_count - mask.shape[1], -1, -1)], dim=1)
        else:
            mask = mask[:, :frame_count]
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask.permute(1, 0, 2, 3), size=(height, width), mode="nearest").permute(1, 0, 2, 3)
    if mask.min() >= 0 and mask.max() <= 1:
        mask = mask * 2 - 1
    return mask.clamp(-1, 1)


def as_batched_5d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor.unsqueeze(0)
    if tensor.ndim == 5:
        return tensor
    raise ValueError(f"Expected a 4D or 5D tensor, got shape {tuple(tensor.shape)}")


def extract_and_compress_mask_to_latent(mask_cthw: torch.Tensor, additional_spatial_downsample: int = 1, temporal_compression_stride: int = 4, label: str = "mask") -> torch.Tensor:
    """Convert a SCAIL-2 RGB mask video in [-1, 1] to 28 binary latent mask channels."""
    if mask_cthw.ndim == 5:
        if mask_cthw.shape[0] != 1:
            raise ValueError(f"Expected a single batched mask, got shape {tuple(mask_cthw.shape)}")
        mask_cthw = mask_cthw[0]
    if mask_cthw.shape[0] == 1:
        mask_cthw = mask_cthw.expand(3, -1, -1, -1)
    if mask_cthw.shape[0] != 3:
        raise ValueError(f"Expected mask channels C=1 or C=3, got shape {tuple(mask_cthw.shape)}")

    _, t, h, w = mask_cthw.shape
    on_threshold = (225.0 - 127.5) / 127.5
    mask = mask_cthw.permute(1, 0, 2, 3).float()
    r = (mask[:, 0:1] > on_threshold).float()
    g = (mask[:, 1:2] > on_threshold).float()
    b = (mask[:, 2:3] > on_threshold).float()
    nr, ng, nb = 1 - r, 1 - g, 1 - b
    binary_7ch = torch.cat([r * g * b, r * ng * nb, nr * g * nb, nr * ng * b, r * g * nb, r * ng * b, nr * g * b], dim=1)

    total = h * w * t
    for idx, name in enumerate(("white", "red", "green", "blue", "yellow", "magenta", "cyan")):
        ratio = binary_7ch[:, idx].sum().item() / total
        if ratio > 0.001:
            logging.info(f"  [SCAIL-2 {label}] ch{idx} {name}: {ratio:.4f} ({ratio * 100:.2f}%)")

    h_lat, w_lat = h, w
    if additional_spatial_downsample > 1:
        h_lat //= additional_spatial_downsample
        w_lat //= additional_spatial_downsample
    for _ in range(3):
        h_lat = (h_lat + 1) // 2
        w_lat = (w_lat + 1) // 2
    binary_7ch = F.interpolate(binary_7ch, size=(h_lat, w_lat), mode="area")

    t_latent = (t - 1) // temporal_compression_stride + 1
    target_t = t_latent * temporal_compression_stride
    padded = torch.cat([binary_7ch[:1].repeat(temporal_compression_stride, 1, 1, 1), binary_7ch[1:]], dim=0)
    if padded.shape[0] < target_t:
        padded = torch.cat([padded, padded[-1:].repeat(target_t - padded.shape[0], 1, 1, 1)], dim=0)
    elif padded.shape[0] > target_t:
        padded = padded[:target_t]
    return padded.view(t_latent, temporal_compression_stride * 7, h_lat, w_lat).permute(1, 0, 2, 3)


SCAIL2_COLOR_BITS = ((True, True, True), (True, False, False), (False, True, False), (False, False, True), (True, True, False), (True, False, True), (False, True, True))


def _is_thwc_video(tensor):
    return torch.is_tensor(tensor) and tensor.ndim == 4 and tensor.shape[-1] in (1, 3, 4)


def _frame_count(tensor):
    return tensor.shape[0] if _is_thwc_video(tensor) else tensor.shape[1]


def _shared_frame_count(*tensors):
    return min(_frame_count(tensor) for tensor in tensors if tensor is not None)


def _video_hw(tensor):
    return (tensor.shape[1], tensor.shape[2]) if _is_thwc_video(tensor) else tensor.shape[-2:]


def _get_hwc_frame(tensor, frame_idx):
    frame = tensor[frame_idx] if _is_thwc_video(tensor) else tensor[:, frame_idx].permute(1, 2, 0)
    if frame.shape[-1] == 1:
        frame = frame.expand(-1, -1, 3)
    elif frame.shape[-1] > 3:
        frame = frame[..., :3]
    return frame


def _frame_to_uint8_hwc(frame):
    if torch.is_tensor(frame):
        frame = frame.detach().cpu()
        if frame.dtype == torch.uint8:
            arr = frame.numpy()
        else:
            frame = frame.float()
            frame = (frame + 1.0) * 127.5 if float(frame.min()) < 0 else frame * 255.0 if float(frame.max()) <= 1 else frame
            arr = frame.clamp(0, 255).to(torch.uint8).numpy()
    else:
        arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def _resize_hwc_uint8(arr, target_h, target_w, crop=False, resample=Image.Resampling.LANCZOS):
    if arr.shape[:2] == (target_h, target_w) and not crop:
        return arr
    img = Image.fromarray(arr, mode="RGB")
    if crop:
        ow, oh = img.size
        if ow / oh > target_w / target_h:
            nw = int(oh * target_w / target_h)
            img = img.crop(((ow - nw) // 2, 0, (ow + nw) // 2, oh))
        else:
            nh = int(ow * target_h / target_w)
            img = img.crop((0, (oh - nh) // 2, ow, (oh + nh) // 2))
    img = img.resize((target_w, target_h), resample=resample)
    return np.asarray(img, dtype=np.uint8)


def _iter_frame_jobs(frame_count, worker, max_workers=1):
    max_workers = max(1, int(max_workers or 1))
    if max_workers == 1 or frame_count <= 1:
        for frame_idx in range(frame_count):
            yield worker(frame_idx)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            next_frame, futures = 0, set()
            while next_frame < frame_count and len(futures) < max_workers:
                futures.add(executor.submit(worker, next_frame))
                next_frame += 1
            while futures:
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    yield future.result()
                    if next_frame < frame_count:
                        futures.add(executor.submit(worker, next_frame))
                        next_frame += 1


def _float_cthw_from_frames(tensor, frame_count, frame_processor, target_h, target_w, max_workers=1):
    output = torch.empty((3, frame_count, target_h, target_w), dtype=torch.float32, device="cpu")

    def process_frame(frame_idx):
        arr = frame_processor(_frame_to_uint8_hwc(_get_hwc_frame(tensor, frame_idx)))
        return frame_idx, torch.from_numpy(np.array(arr, copy=True)).permute(2, 0, 1).to(torch.float32).div_(127.5).sub_(1.0)

    for frame_idx, frame in _iter_frame_jobs(frame_count, process_frame, max_workers=max_workers):
        output[:, frame_idx].copy_(frame)
    return output


def _resize_video_cthw_float(tensor, target_h, target_w, crop=False, max_workers=1, frame_count=None):
    frame_count = _frame_count(tensor) if frame_count is None else min(frame_count, _frame_count(tensor))
    if not _is_thwc_video(tensor) and tensor.dtype != torch.uint8 and tensor.shape[-2:] == (target_h, target_w) and not crop:
        return tensor if frame_count == tensor.shape[1] else tensor[:, :frame_count].contiguous()
    return _float_cthw_from_frames(tensor, frame_count, lambda arr: _resize_hwc_uint8(arr, target_h, target_w, crop=crop, resample=Image.Resampling.LANCZOS), target_h, target_w, max_workers=max_workers)


def _first_frame_to_image(tensor):
    return Image.fromarray(_frame_to_uint8_hwc(_get_hwc_frame(tensor, 0)), mode="RGB") if _is_thwc_video(tensor) or tensor.dtype == torch.uint8 else convert_tensor_to_image(tensor[:, 0])


def _color_selector(arr, color_idx):
    r, g, b = arr[..., 0] > 225, arr[..., 1] > 225, arr[..., 2] > 225
    nr, ng, nb = ~r, ~g, ~b
    return (r & g & b, r & ng & nb, nr & g & nb, nr & ng & b, r & g & nb, r & ng & b, nr & g & b)[color_idx]


def _color_presence(arr):
    return np.asarray([_color_selector(arr, idx).any() for idx in range(len(SCAIL2_COLOR_BITS))], dtype=bool)


def _analyze_single_color_mask(mask_cthw, object_colors, frame_count, max_workers=1):
    if not object_colors:
        return None
    present = np.zeros(len(SCAIL2_COLOR_BITS), dtype=bool)
    for frame_present in _iter_frame_jobs(frame_count, lambda frame_idx: _color_presence(_frame_to_uint8_hwc(_get_hwc_frame(mask_cthw, frame_idx))), max_workers=max_workers):
        present |= frame_present
    if int(present.sum()) != 1:
        return None
    target_bits = tuple(bool(v) for v in (np.asarray(object_colors[0]) >= 128).tolist())
    target_idx = SCAIL2_COLOR_BITS.index(target_bits) if target_bits in SCAIL2_COLOR_BITS else -1
    source_idx = int(np.flatnonzero(present)[0])
    return None if source_idx == target_idx or target_idx < 0 else ("single", source_idx, np.asarray(object_colors[0], dtype=np.uint8))


def _analyze_replace_mask(mask_cthw, object_colors, frame_count, target_h, target_w, crop, max_workers=1):
    black_count, total = 0, 0
    def process_frame(frame_idx):
        arr = _resize_hwc_uint8(_frame_to_uint8_hwc(_get_hwc_frame(mask_cthw, frame_idx)), target_h, target_w, crop=crop, resample=Image.Resampling.NEAREST)
        black = (arr[..., 0] < 30) & (arr[..., 1] < 30) & (arr[..., 2] < 30)
        return int(black.sum()), black.size

    for frame_black_count, frame_total in _iter_frame_jobs(frame_count, process_frame, max_workers=max_workers):
        black_count += frame_black_count
        total += frame_total
    return None if total == 0 or black_count / total <= 0.33 else ("replace", np.asarray((object_colors or [(0, 0, 255)])[0], dtype=np.uint8))


def _apply_mask_normalizer(arr, normalizer):
    if normalizer is None:
        return arr
    if normalizer[0] == "single":
        output = np.zeros_like(arr)
        output[_color_selector(arr, normalizer[1])] = normalizer[2]
        return output
    output = arr.copy()
    white = (arr[..., 0] > 225) & (arr[..., 1] > 225) & (arr[..., 2] > 225)
    black = (arr[..., 0] < 30) & (arr[..., 1] < 30) & (arr[..., 2] < 30)
    output[white] = normalizer[1]
    output[black] = (255, 255, 255)
    return output


def _expand_colored_frame(arr, object_colors, expand_scale, background_color=None):
    if int(expand_scale or 0) == 0 or not object_colors:
        return arr
    colors = np.asarray(object_colors, dtype=np.uint8).reshape(-1, 3)
    background = np.asarray([0, 0, 0] if background_color is None else background_color, dtype=np.uint8).reshape(1, 1, 3)
    output = np.broadcast_to(background, arr.shape).copy()
    occupied = np.zeros(arr.shape[:2], dtype=bool)
    for color in colors:
        selector = np.where(color.reshape(1, 1, 3) >= 128, arr >= 128, arr < 128).all(axis=-1)
        selector = expand_or_shrink_mask(selector.astype(np.uint8) * 255, expand_scale) > 127
        selector &= ~occupied
        output[selector] = color
        occupied |= selector
    return output


def _prepare_scail2_mask_cthw(mask, target_h, target_w, model_def, replace_mode, expand_scale, crop=False, max_workers=1, frame_count=None, resize_first=False):
    if mask is None:
        return None
    object_colors = (model_def or {}).get("magic_mask_object_colors", [])
    frame_count = _frame_count(mask) if frame_count is None else min(frame_count, _frame_count(mask))
    normalizer = _analyze_replace_mask(mask, object_colors, frame_count, target_h, target_w, crop, max_workers=max_workers) if replace_mode else _analyze_single_color_mask(mask, object_colors, frame_count, max_workers=max_workers)
    background_color = (model_def or {}).get("video_mask_replace_background_color" if replace_mode else "video_mask_background_color", None)

    def process_frame(arr):
        if resize_first:
            arr = _resize_hwc_uint8(arr, target_h, target_w, crop=crop, resample=Image.Resampling.NEAREST)
        arr = _apply_mask_normalizer(arr, normalizer)
        arr = _expand_colored_frame(arr, object_colors, expand_scale, background_color=background_color)
        return arr if resize_first else _resize_hwc_uint8(arr, target_h, target_w, crop=crop, resample=Image.Resampling.NEAREST)

    return _float_cthw_from_frames(mask, frame_count, process_frame, target_h, target_w, max_workers=max_workers)


def _expected_reference_color(custom_settings, index, model_def):
    if not isinstance(custom_settings, dict):
        return None
    colors = custom_settings.get("scail2_reference_expected_colors")
    if not isinstance(colors, (list, tuple)) or index >= len(colors):
        return None
    color = colors[index]
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        return None
    try:
        normalized = tuple(min(255, max(0, int(channel))) for channel in color)
    except (TypeError, ValueError):
        return None
    object_colors = (model_def or {}).get("magic_mask_object_colors", [])
    return normalized if not object_colors or normalized in [tuple(item) for item in object_colors] else None


def _active_scail2_reference_colors(
    mask_cthw, expected_colors, minimum_area_ratio=0.0001,
):
    """Return semantic reference colors visibly active in one model window."""
    if mask_cthw is None or not torch.is_tensor(mask_cthw):
        return []
    mask = mask_cthw[0] if mask_cthw.ndim == 5 else mask_cthw
    if mask.ndim != 4 or mask.shape[0] < 3:
        return []
    mask = mask[:3].detach()
    if mask.numel() == 0:
        return []
    if mask.dtype == torch.uint8 or float(mask.max()) > 1.0:
        threshold = 127.5
    elif float(mask.min()) < 0.0:
        threshold = 0.0
    else:
        threshold = 0.5
    high = mask > threshold
    frame_area = int(mask.shape[-2] * mask.shape[-1])
    minimum_pixels = max(
        16,
        int(round(frame_area * max(0.0, float(minimum_area_ratio)))),
    )

    active_colors = []
    seen = set()
    for raw_color in expected_colors or []:
        if raw_color is None:
            continue
        try:
            color = tuple(
                min(255, max(0, int(channel)))
                for channel in raw_color
            )
        except (TypeError, ValueError):
            continue
        if len(color) != 3 or color in seen:
            continue
        seen.add(color)
        bits = tuple(channel >= 128 for channel in color)
        selector = torch.ones(
            mask.shape[1:],
            device=mask.device,
            dtype=torch.bool,
        )
        for channel_index, bit in enumerate(bits):
            selector &= (
                high[channel_index]
                if bit
                else ~high[channel_index]
            )
        per_frame_area = selector.flatten(1).sum(dim=1)
        if (
            per_frame_area.numel()
            and int(per_frame_area.max().item()) >= minimum_pixels
        ):
            active_colors.append(color)
    return active_colors


def _select_scail2_window_reference_indices(
    expected_colors, active_colors,
):
    """Choose reference pairs whose semantic colors occur in this window."""
    colors = []
    for raw_color in expected_colors or []:
        if raw_color is None:
            colors.append(None)
            continue
        try:
            color = tuple(int(channel) for channel in raw_color)
        except (TypeError, ValueError):
            return list(range(len(expected_colors or [])))
        if len(color) != 3:
            return list(range(len(expected_colors or [])))
        colors.append(color)
    all_indices = list(range(len(colors)))
    distinct_colors = {color for color in colors if color is not None}
    if len(distinct_colors) < 2:
        return all_indices

    active = set()
    for raw_color in active_colors or []:
        try:
            color = tuple(int(channel) for channel in raw_color)
        except (TypeError, ValueError):
            continue
        if len(color) == 3:
            active.add(color)
    if not active:
        uncolored_indices = [
            index for index, color in enumerate(colors)
            if color is None
        ]
        return uncolored_indices or all_indices
    if len(active) == 1:
        # A shared cast is the trained path when several mapped people occur
        # together, but it still contains every identity. In a solo-character
        # window that lets a visually stronger inactive cast member leak into
        # the active mask. Promote the matching character's own full reference
        # to primary and omit every uncolored/group reference for that window.
        matching_indices = [
            index
            for index, color in enumerate(colors)
            if color in active
        ]
        if matching_indices:
            return matching_indices
    if len(active) >= 2:
        # The native multi-person path is the single color-mapped cast
        # reference. SCAIL-2's additional-reference path is explicitly
        # experimental; stacking full/detail views for every active character
        # can make the final (or visually strongest) identity take over every
        # semantic color. Keep the shared cast as primary and let the separate
        # timeline scene anchor preserve the surroundings.
        group_indices = [
            index
            for index, color in enumerate(colors)
            if color is None
        ]
        if group_indices:
            return group_indices

    selected = [
        index
        for index, color in enumerate(colors)
        if color is None or color in active
    ]
    return selected or all_indices


def normalize_single_color_mask(mask_cthw: torch.Tensor, model_def, target_color=None) -> torch.Tensor:
    if mask_cthw is None:
        return mask_cthw
    object_colors = (model_def or {}).get("magic_mask_object_colors", [])
    if not object_colors:
        return mask_cthw
    mask = mask_cthw if mask_cthw.shape[0] == 3 else mask_cthw[:1].expand(3, -1, -1, -1)
    on_threshold = (225.0 - 127.5) / 127.5
    off_threshold = (30.0 - 127.5) / 127.5
    active = mask.float().permute(1, 0, 2, 3)
    r = active[:, 0:1] > on_threshold
    g = active[:, 1:2] > on_threshold
    b = active[:, 2:3] > on_threshold
    black = (
        (active[:, 0:1] < off_threshold)
        & (active[:, 1:2] < off_threshold)
        & (active[:, 2:3] < off_threshold)
    )
    white = r & g & b
    # SCAIL-2 multi-reference uses a binary semantic mask for clean scene
    # evidence: white pixels are visible background and black pixels are
    # intentionally hidden. Do not reinterpret that white region as the
    # primary character color.
    binary_background = black | white
    if (
        bool(black.any())
        and bool(white.any())
        and bool(binary_background.all())
    ):
        return mask_cthw

    target_rgb = object_colors[0] if target_color is None else target_color
    target_color = torch.as_tensor(target_rgb, device=mask_cthw.device, dtype=mask_cthw.dtype).view(3, 1, 1, 1) / 127.5 - 1.0
    nr, ng, nb = ~r, ~g, ~b
    color_masks = torch.cat([r & g & b, r & ng & nb, nr & g & nb, nr & ng & b, r & g & nb, r & ng & b, nr & g & b], dim=1)
    present = color_masks.flatten(2).any(dim=2).any(dim=0)
    if int(present.sum().item()) != 1:
        return mask_cthw
    target_on = (torch.as_tensor(target_rgb, device=mask_cthw.device) >= 128).tolist()
    color_bits = [(True, True, True), (True, False, False), (False, True, False), (False, False, True), (True, True, False), (True, False, True), (False, True, True)]
    target_idx = color_bits.index(tuple(bool(v) for v in target_on)) if tuple(bool(v) for v in target_on) in color_bits else -1
    source_idx = int(torch.nonzero(present, as_tuple=False)[0].item())
    if source_idx == target_idx:
        return mask_cthw
    selector = color_masks[:, source_idx:source_idx + 1].permute(1, 0, 2, 3)
    output = torch.full_like(mask, -1)
    return torch.where(selector.expand_as(output), target_color.expand_as(output), output)


def normalize_driving_mask_for_mode(mask_cthw: torch.Tensor, model_def, replace_mode: bool) -> torch.Tensor:
    if mask_cthw is None:
        return None
    if not replace_mode:
        return normalize_single_color_mask(mask_cthw, model_def)
    mask = mask_cthw if mask_cthw.shape[0] == 3 else mask_cthw[:1].expand(3, -1, -1, -1)
    on_threshold = (225.0 - 127.5) / 127.5
    off_threshold = (30.0 - 127.5) / 127.5
    rgb = mask.float().permute(1, 0, 2, 3)
    black = (rgb[:, 0:1] < off_threshold) & (rgb[:, 1:2] < off_threshold) & (rgb[:, 2:3] < off_threshold)
    if black.float().mean().item() <= 0.33:
        return mask_cthw
    white = (rgb[:, 0:1] > on_threshold) & (rgb[:, 1:2] > on_threshold) & (rgb[:, 2:3] > on_threshold)
    output = mask.clone()
    first_color = (model_def or {}).get("magic_mask_object_colors", [(0, 0, 255)])[0]
    blue = torch.as_tensor(first_color, device=mask.device, dtype=mask.dtype).view(3, 1, 1, 1) / 127.5 - 1.0
    if white.any().item():
        output = torch.where(white.permute(1, 0, 2, 3).expand_as(output), blue.expand_as(output), output)
    return torch.where(black.permute(1, 0, 2, 3).expand_as(output), torch.ones_like(output), output)


def _resize_mask_cthw(tensor, target_h, target_w, crop=False):
    if tensor is None or tensor.shape[-2:] == (target_h, target_w):
        return tensor
    c, t, h, w = tensor.shape
    if crop:
        new_h, new_w = (target_h, int(w * target_h / h)) if w / h > target_w / target_h else (int(h * target_w / w), target_w)
    else:
        new_h, new_w = target_h, target_w
    frames = F.interpolate(tensor.permute(1, 0, 2, 3), size=(new_h, new_w), mode="nearest")
    if crop:
        top, left = (new_h - target_h) // 2, (new_w - target_w) // 2
        frames = frames[:, :, top:top + target_h, left:left + target_w]
    return frames.permute(1, 0, 2, 3).contiguous()


def _trim_to_shared_frame_count(*tensors):
    frame_count = min(tensor.shape[1] for tensor in tensors if tensor is not None)
    return tuple(None if tensor is None else tensor[:, :frame_count].contiguous() for tensor in tensors)


def _extract_max_people(video_prompt_type):
    return int(m.group(1)) if (m := re.search(r'(?<!#)([1-5])(?!#)', video_prompt_type or "")) else 1


def preprocess_all_scail2(video_prompt_type=None, custom_settings=None, **kwargs):
    custom_settings = custom_settings if isinstance(custom_settings, dict) else {}
    return not test_scail2_replace(video_prompt_type) and _extract_max_people(video_prompt_type) == 1 and custom_settings.get("scail2_animate_preprocessing", SCAIL2_ANIMATE_PREPROCESSING_RAW) == SCAIL2_ANIMATE_PREPROCESSING_POSE


def custom_preprocess_scail2(video_guide, video_mask, pre_video_guide=None, max_workers=1, expand_scale=0, video_prompt_type=None, **kwargs):
    model_def = kwargs.get("model_def") or {}
    custom_settings = kwargs.get("custom_settings", {})
    if not isinstance(custom_settings, dict):
        custom_settings = {}
    replace_mode = test_scail2_replace(video_prompt_type) or pre_video_guide is None
    ref_image = (
        _first_frame_to_image(video_guide)
        if replace_mode or pre_video_guide is None
        else convert_tensor_to_image(pre_video_guide[:, 0])
    )
    target_w, target_h = int(kwargs.get("width", ref_image.width)), int(kwargs.get("height", ref_image.height))
    fit_crop = kwargs.get("fit_crop", False)
    fit_canvas = kwargs.get("fit_canvas", None)
    source_h, source_w = _video_hw(video_guide)
    if fit_canvas is not None and not fit_crop:
        target_h, target_w = calculate_new_dimensions(
            target_h, target_w, source_h, source_w,
            fit_canvas, kwargs.get("block_size", 16),
        )

    if replace_mode:
        frame_count = _shared_frame_count(video_guide, video_mask)
        frames = _resize_video_cthw_float(video_guide, target_h, target_w, crop=fit_crop, max_workers=max_workers, frame_count=frame_count)
        video_guide = None
        video_mask = _prepare_scail2_mask_cthw(video_mask, target_h, target_w, model_def, replace_mode=True, expand_scale=expand_scale, crop=fit_crop, max_workers=max_workers, frame_count=frame_count, resize_first=True)
        return frames, None, video_mask, None

    pose_mask = kwargs.get("pose_mask", None)
    frame_count = _shared_frame_count(video_guide, video_mask, pose_mask)
    animate_preprocessing = custom_settings.get("scail2_animate_preprocessing", SCAIL2_ANIMATE_PREPROCESSING_RAW)
    if animate_preprocessing != SCAIL2_ANIMATE_PREPROCESSING_POSE:
        frames = _resize_video_cthw_float(video_guide, target_h, target_w, crop=fit_crop, max_workers=max_workers, frame_count=frame_count)
        video_guide = None
        video_mask = _prepare_scail2_mask_cthw(video_mask, target_h, target_w, model_def, replace_mode=False, expand_scale=expand_scale, crop=fit_crop, max_workers=max_workers, frame_count=frame_count)
        return frames, None, video_mask, None

    source_h, source_w = _video_hw(video_guide)
    video_mask = _prepare_scail2_mask_cthw(video_mask, source_h, source_w, model_def, replace_mode=False, expand_scale=expand_scale, max_workers=max_workers, frame_count=frame_count)
    pose_mask = (video_mask + 1.0).mul(0.5).amax(dim=0, keepdim=True).clamp_(0, 1) if video_mask is not None else pose_mask
    video_guide = _resize_video_cthw_float(video_guide, source_h, source_w, max_workers=max_workers, frame_count=frame_count)
    if ref_image.size != (target_w, target_h):
        from PIL import ImageOps
        ref_image = ImageOps.fit(ref_image, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    from ..scail import ScailPoseProcessor

    max_people = _extract_max_people(video_prompt_type)
    processor = ScailPoseProcessor(multi_person=max_people > 1, max_people=max_people)
    video_guide_processed = processor.extract_and_render(video_guide, ref_image=ref_image, mask_frames=pose_mask, align_pose=True)
    if video_guide_processed.numel() == 0:
        import gradio as gr
        gr.Info("Unable to detect a Person")
        return None, None, None, None
    return video_guide_processed, None, video_mask, None


def build_scail2_pose_tokens(
    model,
    pose_latents: torch.Tensor,
    driving_masks: torch.Tensor,
    target_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    pose_latents = as_batched_5d(pose_latents)
    driving_masks = as_batched_5d(driving_masks)
    pose_mask = torch.ones(pose_latents.shape[0], 4, *pose_latents.shape[2:], device=pose_latents.device, dtype=pose_latents.dtype)
    pose_input = torch.cat([pose_latents, pose_mask], dim=1)
    pose_emb = model.pose_patch_embedding(pose_input.to(model.pose_patch_embedding.weight.dtype))
    mask_emb = model.mask_patch_embedding(driving_masks.to(model.mask_patch_embedding.weight.dtype))
    tokens = (pose_emb + mask_emb).flatten(2).transpose(1, 2)
    return tokens if target_dtype is None else tokens.to(target_dtype)


def _downsample_pose_freqs(freqs_cos, freqs_sin, grid_t, grid_h, grid_w):
    head_dim = freqs_cos.shape[1]
    freqs_cos = freqs_cos.view(grid_t, grid_h, grid_w, head_dim).permute(0, 3, 1, 2)
    freqs_sin = freqs_sin.view(grid_t, grid_h, grid_w, head_dim).permute(0, 3, 1, 2)
    freqs_cos = F.avg_pool2d(freqs_cos, kernel_size=2, stride=2).permute(0, 2, 3, 1).reshape(-1, head_dim)
    freqs_sin = F.avg_pool2d(freqs_sin, kernel_size=2, stride=2).permute(0, 2, 3, 1).reshape(-1, head_dim)
    return freqs_cos, freqs_sin


def _tensor_or_image_to_cthw(image, device, dtype):
    if torch.is_tensor(image):
        image = image.to(device=device, dtype=dtype)
        return image.unsqueeze(1) if image.ndim == 3 else image
    return convert_image_to_tensor(image).unsqueeze(1).to(device=device, dtype=dtype)


def _resize_ref_image(image_ref, height, width):
    if image_ref.shape[-2:] == (height, width):
        return image_ref.clone()
    return F.interpolate(image_ref.permute(1, 0, 2, 3), size=(height, width), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)


def _center_crop_ref_image_to_canvas(image_ref, height, width):
    """Scale a reference to fill the control canvas, then center-crop it.

    SCAIL-2's native replacement preprocessing uses a centered rectangle
    crop for reference images. Letterboxing a portrait into a wide video
    canvas makes the person much smaller and weakens the identity signal.
    The control video still owns ``height`` and ``width``; the reference
    never changes the output aspect ratio.
    """
    ref_h, ref_w = image_ref.shape[-2:]
    scale = max(height / ref_h, width / ref_w)
    new_h = max(height, int(round(ref_h * scale)))
    new_w = max(width, int(round(ref_w * scale)))
    resized = _resize_ref_image(image_ref, new_h, new_w)
    top, left = (new_h - height) // 2, (new_w - width) // 2
    return resized[:, :, top:top + height, left:left + width].contiguous()


def _center_crop_ref_mask_to_canvas(ref_mask, height, width):
    """Apply the reference image's centered fill-and-crop to its mask."""
    ref_h, ref_w = ref_mask.shape[-2:]
    scale = max(height / ref_h, width / ref_w)
    new_h = max(height, int(round(ref_h * scale)))
    new_w = max(width, int(round(ref_w * scale)))
    if (new_h, new_w) != (ref_h, ref_w):
        ref_mask = F.interpolate(
            ref_mask.permute(1, 0, 2, 3),
            size=(new_h, new_w),
            mode="nearest",
        ).permute(1, 0, 2, 3)
    top, left = (new_h - height) // 2, (new_w - width) // 2
    return ref_mask[:, :, top:top + height, left:left + width].contiguous()


def _center_crop_ref_alpha_to_canvas(alpha_mask, height, width):
    """Apply the reference crop to a soft alpha matte."""
    ref_h, ref_w = alpha_mask.shape[-2:]
    scale = max(height / ref_h, width / ref_w)
    new_h = max(height, int(round(ref_h * scale)))
    new_w = max(width, int(round(ref_w * scale)))
    if (new_h, new_w) != (ref_h, ref_w):
        alpha_mask = F.interpolate(
            alpha_mask.permute(1, 0, 2, 3),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)
    top, left = (new_h - height) // 2, (new_w - width) // 2
    return alpha_mask[:, :, top:top + height, left:left + width].contiguous()


def _load_reference_alpha(custom_settings, height, width, device, dtype):
    """Recover meaningful PNG alpha before WanGP's normal RGB conversion.

    Reference attachments are intentionally normalized to RGB by the shared
    loader. Recast still keeps the original upload path so a user-supplied
    cutout can retain its soft hair/clothing edges for SCAIL conditioning.
    """
    if not isinstance(custom_settings, dict):
        return None
    raw_path = custom_settings.get("scail2_reference_alpha_path")
    if not raw_path:
        return None
    alpha_path = Path(str(raw_path))
    if not alpha_path.is_file():
        return None
    try:
        with Image.open(alpha_path) as source_image:
            rgba = ImageOps.exif_transpose(source_image).convert("RGBA")
            alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    except Exception as exc:
        logging.warning(f"Could not read SCAIL-2 reference alpha: {exc}")
        return None
    transparent_fraction = (
        float(np.count_nonzero(alpha < 250)) / float(alpha.size)
        if alpha.size else 0.0
    )
    if (
        alpha.size == 0
        or transparent_fraction < 0.02
        or int(alpha.max()) <= 5
    ):
        return None
    alpha_tensor = torch.from_numpy(np.array(alpha, copy=True)).to(
        device=device, dtype=dtype,
    ).div_(255.0).unsqueeze(0).unsqueeze(1)
    if alpha_tensor.shape[-2:] != (height, width):
        alpha_tensor = F.interpolate(
            alpha_tensor.permute(1, 0, 2, 3),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)
    return alpha_tensor.clamp_(0.0, 1.0)


def _load_clip_identity_reference(custom_settings, height, width, device, dtype):
    """Load Recast's original-background primary image for CLIP identity.

    The regular prepared reference remains the VAE/spatial condition. Keeping
    this channel separate lets Maestro neutralize background leakage without
    weakening the face and clothing signal CLIP receives.
    """
    if not isinstance(custom_settings, dict):
        return None
    raw_path = custom_settings.get("scail2_clip_reference_path")
    if not raw_path:
        return None
    identity_path = Path(str(raw_path))
    if not identity_path.is_file():
        logging.warning(
            "SCAIL-2 CLIP identity reference not found; using the spatial "
            f"reference instead: {identity_path}"
        )
        return None
    try:
        with Image.open(identity_path) as source_image:
            identity_image = ImageOps.exif_transpose(source_image).convert("RGB")
            identity_ref = convert_image_to_tensor(identity_image).unsqueeze(1)
    except Exception as exc:
        logging.warning(
            "Could not read SCAIL-2 CLIP identity reference; using the "
            f"spatial reference instead: {exc}"
        )
        return None
    identity_ref = identity_ref.to(device=device, dtype=dtype)
    return _center_crop_ref_image_to_canvas(
        identity_ref, int(height), int(width),
    )


def _load_static_source_scene_reference(
    custom_settings, height, width, device, dtype,
):
    """Load Recast's stable scene/mask pair for upstream-style segmentation."""

    if not isinstance(custom_settings, dict):
        return None
    raw_image_path = custom_settings.get(
        "scail2_source_scene_reference_path",
    )
    raw_mask_path = custom_settings.get("scail2_source_scene_mask_path")
    if not raw_image_path and not raw_mask_path:
        return None
    if not raw_image_path or not raw_mask_path:
        raise ValueError(
            "SCAIL-2 source-scene conditioning requires both its image and "
            "mask paths."
        )
    image_path = Path(str(raw_image_path))
    mask_path = Path(str(raw_mask_path))
    if not image_path.is_file():
        raise ValueError(
            f"SCAIL-2 source-scene reference not found: {image_path}"
        )
    if not mask_path.is_file():
        raise ValueError(
            f"SCAIL-2 source-scene mask not found: {mask_path}"
        )
    with Image.open(image_path) as source_image:
        image = convert_image_to_tensor(
            ImageOps.exif_transpose(source_image).convert("RGB"),
        ).unsqueeze(1)
    with Image.open(mask_path) as source_mask:
        mask = convert_image_to_tensor(
            ImageOps.exif_transpose(source_mask).convert("RGB"),
        ).unsqueeze(1)
    image = image.to(device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=dtype)
    return {
        "image": _center_crop_ref_image_to_canvas(
            image, int(height), int(width),
        ),
        "mask": _center_crop_ref_mask_to_canvas(
            mask, int(height), int(width),
        ),
        "frame_index": 0,
        "hidden_fraction": float((mask < 0).float().mean().item()),
        "expansion_pixels": 0,
        "stable": True,
    }


def _get_scail2_recast_warmup_frames(custom_settings):
    """Read the trusted internal warm-up without accepting unbounded values."""
    if not isinstance(custom_settings, dict):
        return 0
    try:
        return min(
            32,
            max(
                0,
                int(custom_settings.get("scail2_recast_warmup_frames", 0) or 0),
            ),
        )
    except (TypeError, ValueError):
        return 0


def _use_scail2_primary_only_continuation_refs(
    custom_settings, prefix_frames_count,
):
    """Limit long Recast continuations to one stable SCAIL-2 reference.

    Supporting face/detail and scene references are useful for establishing
    the first window, but SCAIL-2's multi-reference path is experimental.
    Reapplying those independent reference streams to every sliding window can
    progressively over-condition the continuation.  Once clean generated
    history exists, retain only the primary reference and that history.
    """
    if (
        not isinstance(custom_settings, dict)
        or custom_settings.get(
            "scail2_primary_only_continuations",
        ) is not True
    ):
        return False
    try:
        return int(prefix_frames_count or 0) > 0
    except (TypeError, ValueError):
        return False


def _get_scail2_identity_latent_reference_index(custom_settings):
    """Select which prepared reference receives the original identity latent."""
    if not isinstance(custom_settings, dict):
        return 0
    try:
        return max(
            0,
            int(
                custom_settings.get(
                    "scail2_identity_latent_reference_index", 0,
                )
                or 0
            ),
        )
    except (TypeError, ValueError):
        return 0


def _semantic_reference_alpha(ref_mask):
    """Convert a colored SCAIL reference mask into foreground opacity."""
    mask = ref_mask if ref_mask.shape[0] == 3 else ref_mask[:1].expand(3, -1, -1, -1)
    off_threshold = (30.0 - 127.5) / 127.5
    rgb = mask.float().permute(1, 0, 2, 3)
    black = (
        (rgb[:, 0:1] < off_threshold)
        & (rgb[:, 1:2] < off_threshold)
        & (rgb[:, 2:3] < off_threshold)
    )
    return (~black).to(dtype=ref_mask.dtype).permute(1, 0, 2, 3)


def _blend_scail2_identity_latents(
    identity_latents, isolated_latents, ref_mask, margin=2,
):
    """Keep the original VAE identity features only around the subject.

    Encoding a pixel-matted reference changes VAE features inside the subject
    as well as outside it, which weakens identity even when CLIP receives the
    untouched image. Encode both versions first, then use the semantic mask in
    latent space: original-reference features stay on the person and the
    neutralized reference remains everywhere else.
    """
    if identity_latents.shape != isolated_latents.shape:
        raise ValueError(
            "SCAIL-2 identity and isolated reference latents must have the "
            f"same shape, got {tuple(identity_latents.shape)} and "
            f"{tuple(isolated_latents.shape)}."
        )
    alpha_frames = _semantic_reference_alpha(ref_mask).permute(1, 0, 2, 3)
    alpha_frames = F.interpolate(
        alpha_frames,
        size=identity_latents.shape[-2:],
        mode="nearest",
    )
    margin = max(0, int(margin or 0))
    if margin > 0:
        kernel = margin * 2 + 1
        alpha_frames = F.max_pool2d(
            alpha_frames, kernel_size=kernel, stride=1, padding=margin,
        )
    latent_frames = int(identity_latents.shape[1])
    if alpha_frames.shape[0] == 1 and latent_frames > 1:
        alpha_frames = alpha_frames.expand(latent_frames, -1, -1, -1)
    elif alpha_frames.shape[0] != latent_frames:
        alpha_frames = alpha_frames[:latent_frames]
    alpha = alpha_frames.permute(1, 0, 2, 3).to(
        device=identity_latents.device,
        dtype=identity_latents.dtype,
    )
    return (
        identity_latents * alpha
        + isolated_latents * (1.0 - alpha)
    )


def _isolate_scail2_reference(image_ref, ref_mask, model_def, alpha_mask=None):
    """Neutralize RGB outside the semantic reference subject.

    SCAIL-2 receives the reference RGB latent and semantic mask as separate
    conditions. Keeping an unrelated room or landscape in the RGB latent can
    therefore leak its color, lighting, or geometry into a replacement even
    when the semantic mask itself is correct.
    """
    opacity = alpha_mask
    if opacity is None:
        opacity = _semantic_reference_alpha(ref_mask)
    opacity = opacity.to(device=image_ref.device, dtype=image_ref.dtype)
    if opacity.shape[1] != image_ref.shape[1]:
        if opacity.shape[1] == 1:
            opacity = opacity.expand(-1, image_ref.shape[1], -1, -1)
        else:
            opacity = opacity[:, :image_ref.shape[1]]
    if opacity.shape[-2:] != image_ref.shape[-2:]:
        opacity = F.interpolate(
            opacity.permute(1, 0, 2, 3),
            size=image_ref.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)
    opacity = opacity.clamp(0.0, 1.0)
    matte_color = (model_def or {}).get(
        "ref_matte_background_color", [127, 127, 127],
    )
    neutral = to_rgb_tensor(
        matte_color, device=image_ref.device, dtype=image_ref.dtype,
    ).view(3, 1, 1, 1).div(127.5).sub(1.0)
    return image_ref * opacity + neutral * (1.0 - opacity)


def _build_dynamic_source_scene_reference(
    pose_pixels, driving_mask_video, frame_index, model_def,
    inpaint_hidden=False,
):
    """Build one clean-background reference from the current control window.

    Recast driving masks use white for the source scene and semantic colors for
    replacement targets. Preserve the current source frame only where its mask
    is white, then hide the old target with a small safety margin. Timeline
    mode inpaints that hole from the surrounding shot; the compatibility path
    retains the historical neutral fill. Refreshing the pair for every sliding
    window prevents a static first-frame background from fighting later cuts.
    """
    if pose_pixels is None or driving_mask_video is None:
        raise ValueError(
            "Dynamic SCAIL-2 scene anchoring needs control frames and masks."
        )
    if pose_pixels.ndim != 4 or driving_mask_video.ndim != 4:
        raise ValueError(
            "Dynamic SCAIL-2 scene anchoring expects C/T/H/W tensors."
        )
    frame_count = min(
        int(pose_pixels.shape[1]),
        int(driving_mask_video.shape[1]),
    )
    if frame_count <= 0:
        raise ValueError(
            "Dynamic SCAIL-2 scene anchoring received no control frames."
        )
    requested_frame_index = min(
        frame_count - 1, max(0, int(frame_index or 0)),
    )
    if pose_pixels.shape[-2:] != driving_mask_video.shape[-2:]:
        raise ValueError(
            "Dynamic SCAIL-2 scene frame and mask dimensions do not match."
        )

    on_threshold = (225.0 - 127.5) / 127.5
    visible_video = (driving_mask_video > on_threshold).all(dim=0)
    flat_visible = visible_video.flatten(1)
    valid_frames = (
        flat_visible.any(dim=1)
        & (~flat_visible).any(dim=1)
    )
    valid_indices = torch.nonzero(
        valid_frames, as_tuple=False,
    ).flatten().tolist()
    if not valid_indices:
        logging.warning(
            "[SCAIL-2] Skipping this window's dynamic source-scene anchor "
            "because no frame contains both the target and visible surroundings."
        )
        return None
    frame_index = min(
        valid_indices,
        key=lambda index: (
            abs(index - requested_frame_index),
            index < requested_frame_index,
            index,
        ),
    )
    scene_frame = pose_pixels[:, frame_index:frame_index + 1]
    visible = visible_video[
        frame_index:frame_index + 1
    ].unsqueeze(0)
    hidden = ~visible

    height, width = scene_frame.shape[-2:]
    expansion = max(2, min(16, int(round(min(height, width) * 0.01))))
    hidden = F.max_pool2d(
        hidden.to(dtype=torch.float32),
        kernel_size=expansion * 2 + 1,
        stride=1,
        padding=expansion,
    ) > 0.5
    visible = (~hidden).to(
        device=scene_frame.device,
        dtype=scene_frame.dtype,
    )
    if not bool(visible.any()):
        logging.warning(
            "[SCAIL-2] Skipping this window's dynamic source-scene anchor "
            "because the replacement fills the usable frame."
        )
        return None
    scene_reference = None
    if inpaint_hidden:
        try:
            import cv2

            source_rgb = (
                scene_frame[:, 0]
                .detach()
                .float()
                .permute(1, 2, 0)
                .clamp(-1.0, 1.0)
                .add(1.0)
                .mul(127.5)
                .round()
                .byte()
                .cpu()
                .numpy()
            )
            inpaint_mask = (
                hidden[0, 0]
                .detach()
                .byte()
                .mul(255)
                .cpu()
                .numpy()
            )
            inpainted_rgb = cv2.inpaint(
                np.ascontiguousarray(source_rgb),
                np.ascontiguousarray(inpaint_mask),
                max(3, expansion * 2),
                cv2.INPAINT_TELEA,
            )
            scene_reference = (
                torch.from_numpy(np.ascontiguousarray(inpainted_rgb))
                .permute(2, 0, 1)
                .unsqueeze(1)
                .to(device=scene_frame.device, dtype=scene_frame.dtype)
                .div(127.5)
                .sub(1.0)
            )
        except Exception as exc:
            logging.warning(
                "[SCAIL-2] Could not inpaint this timeline scene anchor; "
                f"using the neutral fallback: {exc}"
            )
    if scene_reference is None:
        matte_color = (model_def or {}).get(
            "ref_matte_background_color", [127, 127, 127],
        )
        neutral = to_rgb_tensor(
            matte_color,
            device=scene_frame.device,
            dtype=scene_frame.dtype,
        ).view(3, 1, 1, 1).div(127.5).sub(1.0)
        scene_reference = (
            scene_frame * visible
            + neutral * (1.0 - visible)
        )
    binary_mask = torch.where(
        visible.bool().expand(3, -1, -1, -1),
        torch.ones_like(scene_reference),
        -torch.ones_like(scene_reference),
    )
    return {
        "image": scene_reference,
        "mask": binary_mask,
        "frame_index": frame_index,
        "hidden_fraction": float(hidden.float().mean().item()),
        "expansion_pixels": expansion,
    }


def _load_aligned_ref_mask(custom_settings, height, width, device, dtype):
    """Load Maestro's exact first-frame target mask for an aligned edit."""
    if not isinstance(custom_settings, dict):
        return None
    raw_path = custom_settings.get("scail2_reference_mask_path")
    if not raw_path:
        return None
    mask_path = Path(str(raw_path))
    if not mask_path.is_file():
        raise ValueError(f"SCAIL-2 aligned reference mask not found: {mask_path}")
    with Image.open(mask_path) as mask_image:
        mask_tensor = convert_image_to_tensor(mask_image.convert("RGB")).unsqueeze(1)
    mask_tensor = mask_tensor.to(device=device, dtype=dtype)
    return _center_crop_ref_mask_to_canvas(mask_tensor, height, width)


def _load_additional_ref_mask(custom_settings, index, height, width, device, dtype):
    """Load one explicit SCAIL-2 additional-reference semantic mask."""
    if not isinstance(custom_settings, dict):
        return None
    raw_paths = custom_settings.get("scail2_additional_reference_mask_paths")
    if not isinstance(raw_paths, (list, tuple)) or index >= len(raw_paths):
        return None
    raw_path = raw_paths[index]
    if not raw_path:
        return None
    mask_path = Path(str(raw_path))
    if not mask_path.is_file():
        raise ValueError(
            f"SCAIL-2 additional reference mask #{index + 1} not found: {mask_path}"
        )
    with Image.open(mask_path) as mask_image:
        mask_tensor = convert_image_to_tensor(mask_image.convert("RGB")).unsqueeze(1)
    mask_tensor = mask_tensor.to(device=device, dtype=dtype)
    return _center_crop_ref_mask_to_canvas(mask_tensor, height, width)


def _resize_ref_image_for_mode(image_ref, height, width, video_prompt_type, model_def):
    # In both Animate and Replace, the reference supplies identity while
    # the control video owns the output framing. Match SCAIL-2's native
    # centered rectangle crop instead of shrinking portraits with padding.
    return _center_crop_ref_image_to_canvas(image_ref, height, width)


def _save_debug_ref_mask(ref_mask, save_masks=False):
    if not save_masks:
        return
    try:
        mask = ref_mask.detach().float().cpu()
        if mask.ndim == 5:
            mask = mask[0]
        if mask.ndim == 4:
            mask = mask[:, 0]
        if mask.shape[0] == 1:
            mask = mask.expand(3, -1, -1)
        elif mask.shape[0] != 3:
            mask = mask[:1].expand(3, -1, -1)
        if float(mask.min()) < 0:
            mask = (mask + 1.0) * 127.5
        elif float(mask.max()) <= 1.0:
            mask = mask * 255.0
        image = Image.fromarray(mask.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy(), mode="RGB")
        image.save(Path.cwd() / SCAIL2_DEBUG_REF_MASK_PATH)
    except Exception as exc:
        logging.warning(f"Could not save SCAIL-2 debug ref mask: {exc}")


def _save_debug_ref_image(image_ref, path, save_masks=False):
    if not save_masks:
        return
    try:
        image = image_ref.detach().float().cpu()
        if image.ndim == 5:
            image = image[0]
        if image.ndim == 4:
            image = image[:, 0]
        if float(image.min()) < 0:
            image = (image + 1.0) * 127.5
        elif float(image.max()) <= 1.0:
            image = image * 255.0
        image = Image.fromarray(image.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy(), mode="RGB")
        image.save(Path.cwd() / path)
    except Exception as exc:
        logging.warning(f"Could not save SCAIL-2 debug ref image: {exc}")


def _is_full_ref_mask(ref_mask) -> bool:
    if ref_mask is None:
        return False
    ref_mask = ref_mask.detach().float()
    return ref_mask.numel() > 0 and float(ref_mask.min()) > 0.98


def _auto_ref_mask(image_ref, custom_settings, model_def, height, width, device, dtype, max_people=1):
    keyword = (custom_settings or {}).get("image_ref_keyword_content", "") if isinstance(custom_settings, dict) else ""
    if not keyword:
        keyword = "human character"
    from shared import magic_mask

    object_colors = model_def.get("magic_mask_object_colors", []) if isinstance(model_def, dict) else []
    color_count = min(max(1, int(max_people or 1)), len(object_colors))
    ref_image = convert_tensor_to_image(image_ref).convert("RGB")
    frame = np.asarray(ref_image, dtype=np.uint8)[None]

    # Maestro: SAM3's text grounding is brittle with the default
    # "human character" phrase — field report: a clear full-frame photo
    # of a woman matched NOTHING while "person"/"woman" matched at 32%
    # coverage on the same pixels. Cascade through generic person
    # keywords so a weak configured phrase degrades gracefully instead
    # of failing the whole job.
    candidates = [keyword]
    for fallback in ("person", "woman", "man"):
        if fallback not in candidates:
            candidates.append(fallback)

    ref_mask = None
    for candidate in candidates:
        mask = magic_mask.generate_keyword_masks(
            frame,
            candidate,
            no_hole=True,
            colorize_objects=len(object_colors) > 0,
            color_palette=object_colors[:color_count] if len(object_colors) > 0 else None,
            max_colored_objects=color_count if len(object_colors) > 0 else None,
        )[0]
        if bool(mask.any()):
            if candidate != keyword:
                logging.info(f"[SCAIL-2] Image Ref keyword '{keyword}' matched nothing in the reference; using '{candidate}' instead.")
            ref_mask = mask
            break
    if ref_mask is None:
        return None
    mask_image = Image.fromarray(ref_mask.astype(np.uint8, copy=False), mode="RGB") if ref_mask.ndim == 3 else Image.fromarray(ref_mask.astype(np.uint8) * 255, mode="L").convert("RGB")
    return prepare_scail2_mask(convert_image_to_tensor(mask_image).unsqueeze(1), 1, height, width, device, dtype)


def _set_black_mask_background(mask_cthw, color):
    mask = mask_cthw if mask_cthw.shape[0] == 3 else mask_cthw[:1].expand(3, -1, -1, -1)
    off_threshold = (30.0 - 127.5) / 127.5
    rgb = mask.float().permute(1, 0, 2, 3)
    background = (rgb[:, 0:1] < off_threshold) & (rgb[:, 1:2] < off_threshold) & (rgb[:, 2:3] < off_threshold)
    color = to_rgb_tensor(color, device=mask.device, dtype=mask.dtype).view(3, 1, 1, 1) / 127.5 - 1.0
    return torch.where(background.permute(1, 0, 2, 3).expand_as(mask), color.expand_as(mask), mask)


def custom_image_ref_postprocessor_scail2(
    src_ref_images, src_ref_masks, width, height, image_start, image_prompt_type, image_end, video_prompt_type,
    send_cmd, model_def, custom_settings, image_start_tensor=None, pre_video_frame=None,
):
    ref_sources = [] if src_ref_images is None else src_ref_images
    ref_source = ref_sources[0] if len(ref_sources) > 0 else image_start_tensor
    additional_ref_sources = ref_sources[1:]
    if ref_source is None:
        ref_source = pre_video_frame
    if ref_source is None:
        ref_source = image_start
    if ref_source is None:
        raise ValueError("SCAIL-2 needs a Reference Image, Start Image, or Continue Video frame to build the image reference mask.")

    if send_cmd is not None:
        send_cmd("progress", [0, "Building SCAIL-2 Image Reference Masks" if additional_ref_sources else "Building SCAIL-2 Image Reference Mask"])

    raw_image_ref = _tensor_or_image_to_cthw(ref_source, "cpu", torch.float32)
    raw_ref_h, raw_ref_w = raw_image_ref.shape[-2:]
    image_ref = _resize_ref_image_for_mode(
        raw_image_ref, height, width, video_prompt_type, model_def,
    )
    ref_h, ref_w = image_ref.shape[-2:]

    ref_mask = _load_aligned_ref_mask(
        custom_settings, ref_h, ref_w, image_ref.device, image_ref.dtype,
    )
    if ref_mask is None:
        primary_reference_people = _extract_max_people(video_prompt_type)
        if isinstance(custom_settings, dict):
            try:
                primary_reference_people = min(
                    5,
                    max(
                        1,
                        int(custom_settings.get(
                            "scail2_primary_reference_people",
                            primary_reference_people,
                        )),
                    ),
                )
            except (TypeError, ValueError):
                pass
        ref_mask = _auto_ref_mask(
            image_ref, custom_settings, model_def, ref_h, ref_w, image_ref.device, image_ref.dtype,
            max_people=primary_reference_people,
        )
    else:
        logging.info("[SCAIL-2] Using Maestro's aligned first-frame target mask for the primary reference.")
    if ref_mask is None or float(ref_mask.max()) <= 0:
        raise ValueError("SCAIL-2 could not extract the image reference mask. Check Image Ref Keyword content.")

    ref_mask = normalize_single_color_mask(
        ref_mask,
        model_def,
        _expected_reference_color(custom_settings, 0, model_def),
    )
    replace_conditioning = test_scail2_replace(video_prompt_type)
    if not replace_conditioning:
        ref_mask = _set_black_mask_background(ref_mask, [255, 255, 255])

    isolate_reference = (
        isinstance(custom_settings, dict)
        and custom_settings.get("scail2_isolate_reference_background") is True
    )
    if isolate_reference:
        reference_alpha = _load_reference_alpha(
            custom_settings, raw_ref_h, raw_ref_w,
            image_ref.device, image_ref.dtype,
        )
        if reference_alpha is not None:
            reference_alpha = _center_crop_ref_alpha_to_canvas(
                reference_alpha, ref_h, ref_w,
            )
        image_ref = _isolate_scail2_reference(
            image_ref, ref_mask, model_def, alpha_mask=reference_alpha,
        )
        logging.info(
            "[SCAIL-2] Isolated primary reference from its background using "
            + ("the source PNG alpha." if reference_alpha is not None else "its semantic subject mask.")
        )

    prepared_refs = [image_ref, ref_mask]
    for idx, additional_ref_source in enumerate(additional_ref_sources):
        additional_ref = _center_crop_ref_image_to_canvas(_tensor_or_image_to_cthw(additional_ref_source, "cpu", torch.float32), height, width)
        additional_h, additional_w = additional_ref.shape[-2:]
        additional_mask = _load_additional_ref_mask(
            custom_settings, idx, additional_h, additional_w,
            additional_ref.device, additional_ref.dtype,
        )
        if additional_mask is None:
            additional_mask = _auto_ref_mask(
                additional_ref, custom_settings, model_def, additional_h, additional_w, additional_ref.device, additional_ref.dtype,
                max_people=_extract_max_people(video_prompt_type),
            )
        else:
            logging.info(
                f"[SCAIL-2] Using explicit semantic mask for additional reference #{idx + 1}."
            )
        if additional_mask is None or float(additional_mask.max()) <= 0:
            raise ValueError(f"SCAIL-2 could not extract additional image reference mask #{idx + 1}. Check Image Ref Keyword content.")
        additional_mask = normalize_single_color_mask(
            additional_mask,
            model_def,
            _expected_reference_color(custom_settings, idx + 1, model_def),
        )
        if not replace_conditioning:
            additional_mask = _set_black_mask_background(additional_mask, [255, 255, 255])
        if isolate_reference:
            additional_ref = _isolate_scail2_reference(
                additional_ref, additional_mask, model_def,
            )
        prepared_refs += [additional_ref, additional_mask]
    return prepared_refs, None


def get_scail2_seed_generator(pipeline, seed, window_no):
    """Keep one deterministic noise stream across a SCAIL-2 window sequence.

    Official SCAIL-2 creates one ``torch.Generator`` before its segment loop.
    Wan2GP invokes the model once per sliding window, so recreating and seeding
    the generator inside every invocation would replay the first segment's
    noise field for every continuation.  Store the stream on the loaded
    pipeline and reset it only when a new first window begins (or the seed or
    device changes).
    """
    seed_value = int(seed)
    device_key = str(pipeline.device)
    stream = getattr(pipeline, "_scail2_seed_stream", None)
    reset_stream = (
        int(window_no or 0) <= 1
        or not isinstance(stream, dict)
        or stream.get("seed") != seed_value
        or stream.get("device") != device_key
        or stream.get("generator") is None
    )
    if reset_stream:
        generator = torch.Generator(device=pipeline.device)
        generator.manual_seed(seed_value)
        stream = {
            "seed": seed_value,
            "device": device_key,
            "generator": generator,
        }
        pipeline._scail2_seed_stream = stream
        print(
            "[SCAIL-2] Initialized continuous segment noise stream "
            f"(seed={seed_value})."
        )
    else:
        print(
            "[SCAIL-2] Continuing segment noise stream for window "
            f"{int(window_no)}."
        )
    return stream["generator"]


def prepare_scail2_conditioning(
    pipeline,
    *,
    input_frames,
    input_masks,
    input_ref_images,
    input_ref_masks,
    input_video,
    pre_video_frame,
    prefix_frames_count,
    overlapped_latents,
    height,
    width,
    VAE_tile_size,
    enable_RIFLEx,
    video_prompt_type,
    custom_settings,
    model_def,
    ps_t,
    ps_h,
    ps_w,
    save_masks=False,
    dedicated_model=False,
):
    enable_RIFLEx = False
    replace_conditioning = test_scail2_replace(video_prompt_type)
    pose_pixels = input_frames.to(device=pipeline.device, dtype=pipeline.VAE_dtype)
    if input_ref_images is None or len(input_ref_images) < 2:
        raise ValueError("SCAIL-2 expected the prepared image reference and its colored mask as the first two image references.")
    if len(input_ref_images) % 2 != 0:
        raise ValueError(
            "SCAIL-2 expected every image reference to have a paired "
            "semantic mask."
        )

    lat_h, lat_w = (
        height // pipeline.vae_stride[1],
        width // pipeline.vae_stride[2],
    )
    pose_frames = pose_pixels.shape[1]
    lat_t = int((pose_frames - 1) // pipeline.vae_stride[0]) + 1
    driving_mask_video = prepare_scail2_mask(
        input_masks,
        pose_frames,
        height,
        width,
        pipeline.device,
        pipeline.VAE_dtype,
    )

    # Distinct characters belong in one color-mapped primary cast reference.
    # Their full/detail views remain useful supporting evidence, but feeding
    # every identity into every segment lets SCAIL-2's experimental extra-ref
    # path collapse a solo shot toward the last character. Route those views
    # by the semantic colors actually present in the current control window.
    reference_pair_count = len(input_ref_images) // 2
    raw_reference_colors = (
        custom_settings.get("scail2_reference_expected_colors")
        if isinstance(custom_settings, dict)
        else None
    )
    valid_reference_colors = (
        isinstance(raw_reference_colors, (list, tuple))
        and len(raw_reference_colors) == reference_pair_count
    )
    reference_colors = []
    if valid_reference_colors:
        for reference_index, raw_color in enumerate(raw_reference_colors):
            if raw_color is None:
                reference_colors.append(None)
                continue
            normalized_color = _expected_reference_color(
                custom_settings,
                reference_index,
                model_def,
            )
            if normalized_color is None:
                valid_reference_colors = False
                break
            reference_colors.append(normalized_color)
    if (
        dedicated_model
        and valid_reference_colors
        and len({
            color for color in reference_colors
            if color is not None
        }) >= 2
    ):
        active_reference_colors = _active_scail2_reference_colors(
            driving_mask_video,
            reference_colors,
        )
        selected_reference_indices = (
            _select_scail2_window_reference_indices(
                reference_colors,
                active_reference_colors,
            )
        )
        if selected_reference_indices != list(range(reference_pair_count)):
            routed_ref_images = []
            for reference_index in selected_reference_indices:
                pair_start = reference_index * 2
                routed_ref_images.extend(
                    input_ref_images[pair_start:pair_start + 2],
                )
            input_ref_images = routed_ref_images
            custom_settings = dict(custom_settings)
            custom_settings["scail2_reference_expected_colors"] = [
                raw_reference_colors[index]
                for index in selected_reference_indices
            ]
            active_summary = ", ".join(
                f"rgb{tuple(color)}"
                for color in active_reference_colors
            ) or "no mapped character"
            print(
                "[SCAIL-2] Routed this window's character references "
                f"({active_summary}): kept "
                f"{len(selected_reference_indices)}/"
                f"{reference_pair_count} pair(s), slots "
                + ", ".join(
                    str(index + 1)
                    for index in selected_reference_indices
                )
                + "."
            )

    image_ref = _tensor_or_image_to_cthw(input_ref_images[0], pipeline.device, pipeline.VAE_dtype)
    # The generalized WanModel needed an unmasked CLIP image plus localized
    # VAE-latent restoration to retain identity after Maestro neutralized a
    # reference background.  The official SCAIL-2 transformer already
    # separates primary/additional mask-token regions; applying those legacy
    # reinforcements over-conditions it and compounds contrast across long
    # segments. Keep them only on the legacy fallback path.
    clip_identity_ref = (
        None
        if dedicated_model
        else _load_clip_identity_reference(
            custom_settings,
            height,
            width,
            pipeline.device,
            pipeline.VAE_dtype,
        )
    )
    identity_reference_loaded = clip_identity_ref is not None
    if clip_identity_ref is None:
        clip_identity_ref = image_ref
    if (
        dedicated_model
        and isinstance(custom_settings, dict)
        and (
            custom_settings.get("scail2_clip_reference_path")
            or custom_settings.get("scail2_identity_latent_isolation") is True
        )
    ):
        logging.info(
            "[SCAIL-2] Dedicated transformer is using upstream reference "
            "conditioning; legacy CLIP/VAE identity reinforcement is "
            "reserved for the generalized-model fallback."
        )
    ref_mask = _tensor_or_image_to_cthw(input_ref_images[1], pipeline.device, pipeline.VAE_dtype)
    additional_ref_pairs = input_ref_images[2:]
    if len(additional_ref_pairs) % 2 != 0:
        raise ValueError("SCAIL-2 expected additional image references to be prepared as image/mask pairs.")
    has_continuation_history = (
        overlapped_latents is not None or input_video is not None
    )
    primary_only_continuation = (
        not dedicated_model
        and
        _use_scail2_primary_only_continuation_refs(
            custom_settings, prefix_frames_count,
        )
        and has_continuation_history
    )
    supporting_reference_count = len(additional_ref_pairs) // 2
    dynamic_scene_reference_requested = (
        isinstance(custom_settings, dict)
        and custom_settings.get(
            "scail2_dynamic_source_scene_reference",
        ) is True
    )
    timeline_scene_reference = (
        isinstance(custom_settings, dict)
        and custom_settings.get(
            "scail2_timeline_source_scene_reference",
        ) is True
    )
    if primary_only_continuation:
        additional_ref_pairs = []
        skipped_parts = []
        if supporting_reference_count:
            skipped_parts.append(
                f"{supporting_reference_count} supporting reference(s)"
            )
        if dynamic_scene_reference_requested:
            skipped_parts.append("the dynamic scene anchor")
        skipped_summary = (
            " and ".join(skipped_parts)
            if skipped_parts
            else "experimental supporting references"
        )
        logging.info(
            "[SCAIL-2] Continuation window is using the primary reference "
            f"plus clean generated history; skipping {skipped_summary}."
        )

    ref_mask = prepare_scail2_mask(ref_mask, 1, height, width, pipeline.device, pipeline.VAE_dtype)
    ref_mask = normalize_single_color_mask(
        ref_mask,
        model_def,
        _expected_reference_color(custom_settings, 0, model_def),
    )
    _save_debug_ref_mask(ref_mask, save_masks=save_masks)

    _save_debug_ref_image(image_ref, SCAIL2_DEBUG_MATTED_REF_PATH, save_masks=save_masks)

    isolated_ref_latents = pipeline.vae.encode([image_ref], VAE_tile_size)[0]
    use_identity_latent_isolation = (
        not dedicated_model
        and
        identity_reference_loaded
        and isinstance(custom_settings, dict)
        and custom_settings.get("scail2_identity_latent_isolation") is True
    )
    identity_latent_reference_index = (
        _get_scail2_identity_latent_reference_index(custom_settings)
    )
    identity_ref_latents = None

    def _identity_latents():
        nonlocal identity_ref_latents
        if identity_ref_latents is None:
            identity_ref_latents = pipeline.vae.encode(
                [clip_identity_ref], VAE_tile_size,
            )[0]
        return identity_ref_latents

    if (
        use_identity_latent_isolation
        and identity_latent_reference_index == 0
    ):
        isolated_ref_latents = _blend_scail2_identity_latents(
            _identity_latents(),
            isolated_ref_latents,
            ref_mask,
        )
        logging.info(
            "[SCAIL-2] Preserving original VAE identity features inside "
            "reference slot 0 while neutralizing its surroundings."
        )
    ref_latents = isolated_ref_latents.unsqueeze(0)
    additional_ref_count = 0
    additional_ref_latents = []
    additional_ref_mask_latents = []
    for idx in range(0, len(additional_ref_pairs), 2):
        additional_ref = _tensor_or_image_to_cthw(additional_ref_pairs[idx], pipeline.device, pipeline.VAE_dtype)
        additional_mask = _tensor_or_image_to_cthw(additional_ref_pairs[idx + 1], pipeline.device, pipeline.VAE_dtype)
        additional_mask = prepare_scail2_mask(additional_mask, 1, height, width, pipeline.device, pipeline.VAE_dtype)
        additional_mask = normalize_single_color_mask(
            additional_mask,
            model_def,
            _expected_reference_color(
                custom_settings, idx // 2 + 1, model_def,
            ),
        )
        additional_latents = pipeline.vae.encode(
            [additional_ref], VAE_tile_size,
        )[0]
        reference_index = idx // 2 + 1
        if (
            use_identity_latent_isolation
            and identity_latent_reference_index == reference_index
        ):
            additional_latents = _blend_scail2_identity_latents(
                _identity_latents(),
                additional_latents,
                additional_mask,
            )
            logging.info(
                "[SCAIL-2] Preserving original VAE identity features inside "
                f"reference slot {reference_index} while neutralizing its "
                "surroundings."
            )
        additional_ref_latents.append(additional_latents)
        additional_ref_mask_latents.append(extract_and_compress_mask_to_latent(additional_mask, additional_spatial_downsample=1, label=f"additional ref mask {idx // 2 + 1}").to(device=pipeline.device, dtype=pipeline.VAE_dtype))
    dynamic_scene_reference = (
        dynamic_scene_reference_requested
        and not primary_only_continuation
    )
    if dynamic_scene_reference:
        scene_reference = None
        if dedicated_model and not timeline_scene_reference:
            scene_reference = _load_static_source_scene_reference(
                custom_settings,
                height,
                width,
                pipeline.device,
                pipeline.VAE_dtype,
            )
        if scene_reference is None:
            # Compatibility fallback for older saved jobs that predate the
            # persisted scene paths. Cache the first usable pair so the
            # dedicated reference stack remains invariant across segments.
            cache_name = "_scail2_dedicated_source_scene_reference"
            if (
                dedicated_model
                and not timeline_scene_reference
                and prefix_frames_count == 0
            ):
                setattr(pipeline, cache_name, None)
            cached_scene_reference = (
                getattr(pipeline, cache_name, None)
                if (
                    dedicated_model
                    and not timeline_scene_reference
                    and prefix_frames_count > 0
                )
                else None
            )
            if isinstance(cached_scene_reference, dict):
                scene_reference = {
                    key: (
                        value.to(
                            device=pipeline.device,
                            dtype=pipeline.VAE_dtype,
                        )
                        if torch.is_tensor(value)
                        else value
                    )
                    for key, value in cached_scene_reference.items()
                }
            else:
                # Window one contains the disposable reverse-motion pre-roll.
                # Later guide tensors begin at the current control boundary.
                scene_frame_index = (
                    _get_scail2_recast_warmup_frames(custom_settings)
                    if prefix_frames_count == 0
                    else 0
                )
                scene_reference = _build_dynamic_source_scene_reference(
                    pose_pixels,
                    driving_mask_video,
                    scene_frame_index,
                    model_def,
                    inpaint_hidden=timeline_scene_reference,
                )
                if (
                    dedicated_model
                    and not timeline_scene_reference
                    and scene_reference is not None
                ):
                    setattr(
                        pipeline,
                        cache_name,
                        {
                            key: (
                                value.detach().cpu()
                                if torch.is_tensor(value)
                                else value
                            )
                            for key, value in scene_reference.items()
                        },
                    )
        if scene_reference is not None:
            scene_latents = pipeline.vae.encode(
                [scene_reference["image"]], VAE_tile_size,
            )[0]
            scene_mask_latents = extract_and_compress_mask_to_latent(
                scene_reference["mask"],
                additional_spatial_downsample=1,
                label="dynamic scene ref mask",
            ).to(device=pipeline.device, dtype=pipeline.VAE_dtype)
            additional_ref_latents.insert(0, scene_latents)
            additional_ref_mask_latents.insert(0, scene_mask_latents)
            if dedicated_model and not timeline_scene_reference:
                logging.info(
                    "[SCAIL-2] Reused stable source-scene reference "
                    f"(hidden={scene_reference['hidden_fraction']:.1%})."
                )
            else:
                logging.info(
                    "[SCAIL-2] Refreshed timeline source-scene anchor from control "
                    f"frame {scene_reference['frame_index']} "
                    f"(hidden={scene_reference['hidden_fraction']:.1%}, "
                    f"margin={scene_reference['expansion_pixels']}px)."
                )
    if additional_ref_latents:
        additional_ref_count = sum(latent.shape[1] for latent in additional_ref_latents)
        if not dedicated_model:
            ref_latents = torch.cat(
                additional_ref_latents + [ref_latents[0]], dim=1,
            ).unsqueeze(0)

    history_latents = None
    expected_history_lat_t = int((prefix_frames_count - 1) // pipeline.vae_stride[0]) + 1 if prefix_frames_count > 0 else 0
    if dedicated_model and prefix_frames_count > 0 and input_video is not None:
        # Match upstream SCAIL-2's segment handoff exactly: retain the last
        # rendered pixel frames, then VAE-encode them again for the next
        # segment.  Reusing the previous segment's raw denoised latent slice
        # looks equivalent, but compounds high-frequency artifacts over long
        # sliding-window generations.
        history_frames = input_video[:, :prefix_frames_count].to(device=pipeline.device, dtype=pipeline.VAE_dtype)
        history_latents = pipeline.vae.encode([history_frames], VAE_tile_size)[0].unsqueeze(0)
        print(
            "[SCAIL-2] Dedicated continuation re-encoded "
            f"{history_frames.shape[1]} rendered history frames as "
            f"{history_latents.shape[2]} latent frames."
        )
    elif overlapped_latents is not None and (expected_history_lat_t == 0 or overlapped_latents.shape[2] >= expected_history_lat_t):
        history_latents = overlapped_latents.to(device=pipeline.device, dtype=ref_latents.dtype)
        if dedicated_model:
            print(
                "[SCAIL-2] Dedicated continuation did not receive rendered "
                "history frames; using the legacy latent-slice fallback."
            )
    elif prefix_frames_count > 0 and input_video is not None:
        history_frames = input_video[:, :prefix_frames_count].to(device=pipeline.device, dtype=pipeline.VAE_dtype)
        history_latents = pipeline.vae.encode([history_frames], VAE_tile_size)[0].unsqueeze(0)
    history_lat_t = 0
    color_reference_frame = None
    history_mask = torch.zeros(4, lat_t, lat_h, lat_w, device=pipeline.device, dtype=pipeline.VAE_dtype)
    if history_latents is not None:
        history_lat_t = min(history_latents.shape[2], lat_t)
        history_latents = history_latents[:, :, :history_lat_t]
        extended_overlapped_latents = history_latents
        history_mask[:, :history_lat_t] = 1
        color_reference_frame = input_video[:, :1].to(device=pipeline.device, dtype=pipeline.VAE_dtype) if input_video is not None else image_ref
    else:
        extended_overlapped_latents = None

    pose_pixels_ds = pose_pixels.permute(1, 0, 2, 3)
    pose_pixels_ds = F.interpolate(pose_pixels_ds, size=(max(1, height // 2), max(1, width // 2)), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    pose_latents = pipeline.vae.encode([pose_pixels_ds], VAE_tile_size)[0].unsqueeze(0)

    driving_mask_video = F.interpolate(driving_mask_video.permute(1, 0, 2, 3), size=(max(1, height // 2), max(1, width // 2)), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    driving_masks = extract_and_compress_mask_to_latent(driving_mask_video, additional_spatial_downsample=1, label="driving mask").to(device=pipeline.device, dtype=pipeline.VAE_dtype).unsqueeze(0)

    ref_mask_latent_28ch = extract_and_compress_mask_to_latent(ref_mask, additional_spatial_downsample=1, label="ref mask").to(device=pipeline.device, dtype=pipeline.VAE_dtype)
    dedicated_additional_ref_latents = None
    dedicated_additional_ref_masks = None
    if dedicated_model and additional_ref_latents:
        dedicated_additional_ref_latents = torch.cat(
            additional_ref_latents, dim=1,
        ).unsqueeze(0)
        dedicated_additional_ref_masks = torch.cat(
            additional_ref_mask_latents, dim=1,
        ).unsqueeze(0)
    if dedicated_model:
        ref_mask_latents = ref_mask_latent_28ch
    else:
        ref_mask_latents = (
            torch.cat(
                additional_ref_mask_latents + [ref_mask_latent_28ch],
                dim=1,
            )
            if additional_ref_mask_latents
            else ref_mask_latent_28ch
        )
    null_noisy_mask = torch.zeros(ref_mask_latents.shape[0], lat_t, lat_h, lat_w, device=pipeline.device, dtype=ref_mask_latents.dtype)
    ref_masks = torch.cat([ref_mask_latents, null_noisy_mask], dim=1).unsqueeze(0)

    main_grid_h = lat_h // ps_h
    main_grid_w = lat_w // ps_w
    pose_grid_t = pose_latents.shape[2] // ps_t
    if additional_ref_count:
        if replace_conditioning:
            ref_freqs_cos, ref_freqs_sin = get_nd_rotary_pos_embed((0, 120, 0), (additional_ref_count + 1, 120 + main_grid_h, main_grid_w), (additional_ref_count + 1, main_grid_h, main_grid_w), L_test=lat_t, enable_riflex=enable_RIFLEx)
            video_freqs_cos, video_freqs_sin = get_nd_rotary_pos_embed((additional_ref_count, 0, 0), (additional_ref_count + lat_t, main_grid_h, main_grid_w), (lat_t, main_grid_h, main_grid_w), L_test=lat_t, enable_riflex=enable_RIFLEx)
            main_freqs_cos, main_freqs_sin = torch.cat([ref_freqs_cos, video_freqs_cos]), torch.cat([ref_freqs_sin, video_freqs_sin])
            pose_freqs_cos, pose_freqs_sin = get_nd_rotary_pos_embed((additional_ref_count, 0, 120), (additional_ref_count + pose_grid_t, main_grid_h, 120 + main_grid_w), (pose_grid_t, main_grid_h, main_grid_w), L_test=lat_t, enable_riflex=enable_RIFLEx)
        else:
            main_grid_t = additional_ref_count + 1 + lat_t
            pose_start_t = additional_ref_count + 1
            main_freqs_cos, main_freqs_sin = get_nd_rotary_pos_embed((0, 0, 0), (main_grid_t, main_grid_h, main_grid_w), (main_grid_t, main_grid_h, main_grid_w), L_test=lat_t, enable_riflex=enable_RIFLEx)
            pose_freqs_cos, pose_freqs_sin = get_nd_rotary_pos_embed((pose_start_t, 0, 120), (pose_start_t + pose_grid_t, main_grid_h, 120 + main_grid_w), (pose_grid_t, main_grid_h, main_grid_w), L_test=lat_t, enable_riflex=enable_RIFLEx)
    else:
        main_grid_t = 1 + lat_t
        if replace_conditioning:
            ref_freqs_cos, ref_freqs_sin = get_nd_rotary_pos_embed((0, 120, 0), (1, 120 + main_grid_h, main_grid_w), (1, main_grid_h, main_grid_w), L_test=lat_t, enable_riflex=enable_RIFLEx)
            video_freqs_cos, video_freqs_sin = get_nd_rotary_pos_embed((0, 0, 0), (lat_t, main_grid_h, main_grid_w), (lat_t, main_grid_h, main_grid_w), L_test=lat_t, enable_riflex=enable_RIFLEx)
            main_freqs_cos, main_freqs_sin = torch.cat([ref_freqs_cos, video_freqs_cos]), torch.cat([ref_freqs_sin, video_freqs_sin])
            pose_freqs_cos, pose_freqs_sin = get_nd_rotary_pos_embed((0, 0, 120), (pose_grid_t, main_grid_h, 120 + main_grid_w), (pose_grid_t, main_grid_h, main_grid_w), L_test=lat_t, enable_riflex=enable_RIFLEx)
        else:
            main_freqs_cos, main_freqs_sin = get_nd_rotary_pos_embed((0, 0, 0), (main_grid_t, main_grid_h, main_grid_w), (main_grid_t, main_grid_h, main_grid_w), L_test=lat_t, enable_riflex=enable_RIFLEx)
            pose_freqs_cos, pose_freqs_sin = get_nd_rotary_pos_embed((1, 0, 120), (1 + pose_grid_t, main_grid_h, 120 + main_grid_w), (pose_grid_t, main_grid_h, main_grid_w), L_test=lat_t, enable_riflex=enable_RIFLEx)
    pose_freqs_cos, pose_freqs_sin = _downsample_pose_freqs(pose_freqs_cos, pose_freqs_sin, pose_grid_t, main_grid_h, main_grid_w)

    if dedicated_model:
        model_kwargs = {
            "scail2_ref_latents": ref_latents,
            "scail2_pose_latents": pose_latents,
            "scail2_driving_masks": driving_masks,
            "scail2_ref_masks": ref_masks,
            "scail2_history_mask": (
                history_mask.unsqueeze(0) if history_lat_t > 0 else None
            ),
            "scail2_additional_ref_latents": (
                dedicated_additional_ref_latents
            ),
            "scail2_additional_ref_masks": dedicated_additional_ref_masks,
            "scail2_replace_flag": replace_conditioning,
        }
        model_freqs = None
    else:
        model_kwargs = {
            "y": history_mask,
            "scail2_ref_latents": ref_latents,
            "scail2_pose_latents": pose_latents,
            "scail2_driving_masks": driving_masks,
            "scail2_ref_masks": ref_masks,
        }
        model_freqs = (
            torch.cat([main_freqs_cos, pose_freqs_cos]),
            torch.cat([main_freqs_sin, pose_freqs_sin]),
        )

    return {
        "kwargs": model_kwargs,
        "freqs": model_freqs,
        "clip_image_start": clip_identity_ref.squeeze(1),
        "extended_overlapped_latents": extended_overlapped_latents,
        "color_reference_frame": color_reference_frame,
        "lat_frames": lat_t,
        "post_decode_pre_trim": (
            _get_scail2_recast_warmup_frames(custom_settings)
            if prefix_frames_count == 0
            else 0
        ),
    }
