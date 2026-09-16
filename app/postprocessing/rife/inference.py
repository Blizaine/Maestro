import os
import torch
from torch.nn import functional as F
# from .model.pytorch_msssim import ssim_matlab
from .ssim import ssim_matlab

from .RIFE_HDv3 import Model as ModelV3
from .RIFE_V4 import Model as ModelV4

def get_frame(frames, frame_no):
    if frame_no >= frames.shape[1]:
        return None
    frame = frames[:, frame_no]
    if frame.dtype == torch.uint8:
        frame = frame.float().div_(255.0)
    else:
        frame = (frame + 1) / 2
        frame = frame.clip(0., 1.)
    return frame

def add_frame(frames, frame, h, w):
    frame = (frame * 2) - 1
    frame = frame.clip(-1., 1.)    
    frame = frame.squeeze(0)
    frame = frame[:, :h, :w]
    frame = frame.unsqueeze(1)
    frames.append(frame.cpu())

def process_frames(model, device, frames, exp=0, *, multiplier=None,
                   abort_callback=None, progress_callback=None):
    """Interpolate each adjacent pair without replacing original frames.

    ``exp`` remains compatible with existing x2/x4 callers. RIFE v4 supports
    arbitrary timesteps, allowing x3 at exactly one third and two thirds.
    """
    factor = multiplier if multiplier is not None else 2 ** exp
    if factor not in (1, 2, 3, 4):
        raise ValueError("RIFE supports x2, x3 and x4 interpolation")
    factor = int(factor)
    supports_timestep = getattr(model, "supports_timestep", False)
    if factor == 3 and not supports_timestep:
        raise ValueError("RIFE x3 requires RIFE v4.26; select RIFE v4")
    if frames.ndim != 4 or frames.shape[1] < 1:
        raise ValueError("RIFE expects [channels, frames, height, width] with at least one frame")
    pos = 0
    output_frames = []

    lastframe = get_frame(frames, 0)
    _,  h, w = lastframe.shape
    scale = 1
    fp16 = False
    supports_timestep = getattr(model, "supports_timestep", False)
    pad_mod = getattr(model, "pad_mod", 32)

    def make_inference(I0, I1, n):
        if n <= 0:
            return []
        if supports_timestep:
            return [model.inference(I0, I1, (i + 1) / (n + 1), scale) for i in range(n)]
        middle = model.inference(I0, I1, scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    tmp = max(pad_mod, int(pad_mod / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    def pad_image(img):
        if(fp16):
            return F.pad(img, padding).half()
        else:
            return F.pad(img, padding)

    I1 = lastframe.to(device, non_blocking=True).unsqueeze(0)
    I1 = pad_image(I1)
    for pos in range(1, frames.shape[1]):
        if abort_callback is not None and abort_callback():
            return None
        frame = get_frame(frames, pos)
        I0 = I1
        I1 = frame.to(device, non_blocking=True).unsqueeze(0)
        I1 = pad_image(I1)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        if ssim < 0.2 or ssim > 0.996:
            # Hold across scene cuts or duplicates; never synthesize over a
            # source frame, including in long videos beyond frame 100.
            output = [I0] * (factor - 1)
        else:
            output = make_inference(I0, I1, factor - 1)

        add_frame(output_frames, lastframe, h, w)
        for mid in output:
            add_frame(output_frames, mid, h, w)
        lastframe = frame
        if progress_callback is not None:
            progress_callback("RIFE interpolation", pos, frames.shape[1] - 1)

    add_frame(output_frames, lastframe, h, w)
    return torch.cat( output_frames, dim=1)

def temporal_interpolation(model_path, frames, exp=0, device="cuda", rife_version="v3",
                           *, multiplier=None, abort_callback=None, progress_callback=None):

    input_was_uint8 = frames.dtype == torch.uint8
    if rife_version == "v4":
        model = ModelV4()
    else:
        model = ModelV3()
    model.load_model(model_path, -1, device=device)

    model.eval()
    model.to(device=device)

    with torch.no_grad():    
        output = process_frames(model, device, frames, exp, multiplier=multiplier,
                                abort_callback=abort_callback, progress_callback=progress_callback)

    if output is not None and input_was_uint8:
        output = output.add_(1.0).mul_(127.5).clamp_(0, 255).to(torch.uint8)
    return output
