"""On-demand H3 VAE work for portable characters; called under the GPU lock."""
from __future__ import annotations

import gc
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def video_codec():
    import torch
    from mmgp import offload
    from shared.utils import files_locator as fl, offload_registry
    from models.minimax_h3.minimax_h3_main import _load_video_vae

    if not torch.cuda.is_available():
        raise ValueError("Encoding or previewing an H3 RefMod requires an NVIDIA GPU. Characters with embedded previews can still be imported without encoding.")
    filename = None
    for name in ("minimax_h3_video_vae_fp16.safetensors", "minimax_h3_video_vae_int8_convrot.safetensors"):
        candidate = fl.locate_file(f"minimax_h3/vae/{name}", error_if_none=False)
        if candidate and Path(candidate).is_file():
            filename = candidate
            break
    if not filename:
        raise ValueError("Install an H3 model's video VAE before exporting or previewing RefMods.")
    vae = _load_video_vae(filename)
    manager = None
    try:
        manager = offload.profile({"vae": vae}, profile_no=3, quantizeTransformer=False,
                                 convertWeightsFloatTo=None, pinnedMemory=False, verboseLevel=-1)
        offload_registry.register_offloadobj("Character RefMod VAE", manager)
        manager.gpu_load("vae")
        with torch.inference_mode():
            yield vae
    finally:
        if manager is not None:
            offload_registry.unregister_offloadobj("Character RefMod VAE", manager)
            manager.release()
        del vae
        gc.collect()
        torch.cuda.empty_cache()


def _fit_image(image, max_edge=1024):
    from PIL import Image

    scale = min(1.0, max_edge / max(image.size))
    width, height = (max(32, int(n * scale) // 32 * 32) for n in image.size)
    return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def encode_character_visual(path: Path, kind: str):
    """Encode an image, or up to eight independent character views from video."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from models.minimax_h3.minimax_h3_main import VIDEO_LATENTS_MEAN, VIDEO_LATENTS_STD
    from models.minimax_h3.packing import MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD, MINIMAX_H3_KEYFRAME_ENCODE_SEED

    if kind == "image":
        with Image.open(path) as image:
            views = [_fit_image(ImageOps.exif_transpose(image))]
    else:
        import cv2
        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or count < 1:
                raise ValueError("Could not read the character video.")
            last = min(count - 1, int(fps * 15) - 1)
            views = []
            for position in np.linspace(0, max(0, last), min(8, count)).round().astype(int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                ok, frame = capture.read()
                if ok:
                    views.append(_fit_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        finally:
            capture.release()
        if len(views) < 2:
            raise ValueError("A character video needs at least two readable reference views.")
    mean = torch.tensor(VIDEO_LATENTS_MEAN).view(1, 24, 1, 1, 1)
    std = torch.tensor(VIDEO_LATENTS_STD).view(1, 24, 1, 1, 1)
    latents = []
    with video_codec() as vae:
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device="cuda").view(1, 3, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device="cuda").view(1, 3, 1, 1, 1)
        for view in views:
            view = view.resize(views[0].size)
            pixels = torch.from_numpy(np.array(view)).permute(2, 0, 1)[None, :, None].to("cuda").float() / 255
            moments = vae._encode_clip((pixels - pixel_mean) / pixel_std)
            raw = DiagonalGaussianDistribution(moments).sample(
                generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
            # Match Maestro's reference encoding before per-generation noise augmentation.
            raw = raw.to(torch.float16).float().cpu()
            latents.append(((raw - mean) / std).to(torch.float16))
    return torch.cat(latents, dim=2).contiguous()


def recover_refmod_images(latent, destination: Path, metadata: dict, update=lambda _: None) -> dict:
    """Decode every stored view at its native spatial grid into lossless PNGs.

    Never resample or overwrite the conditioning latent. Stacked image refs
    represent independent pictures, not a continuous video to decode together.
    Unknown/mixed temporal layouts retain the old representative-view behavior
    and are identified as such rather than promising original video frames.
    """
    import re
    import torch
    from PIL import Image
    from models.minimax_h3.minimax_h3_main import VIDEO_LATENTS_MEAN, VIDEO_LATENTS_STD
    from models.minimax_h3.packing import MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD
    from .character_views import MAX_VIEWS, MAX_PIXELS

    count = int(latent.shape[2])
    if not 1 <= count <= MAX_VIEWS or latent.shape[-2] * latent.shape[-1] * 256 > MAX_PIXELS:
        raise ValueError("This RefMod exceeds the character recovery size limit.")
    shapes = re.findall(r"(\d+)x(\d+)x(\d+)", str(metadata.get("source_shape", "")))
    independent = (count == 1 or metadata.get("view_layout") == "independent_images"
                   or bool(shapes) and all(int(shape[0]) == 1 for shape in shapes))
    notes = []
    if metadata.get("mode") in {"pooled", "training"}:
        notes.append("This RefMod stores compressed views. PNG recovery cannot restore detail removed during compression.")
    if not independent:
        notes.append("These are representative latent views. Video or mixed-reference frame boundaries may not be recoverable.")
    items = []
    with video_codec() as vae:
        device = next(vae.parameters()).device
        mean = torch.tensor(VIDEO_LATENTS_MEAN, device=device).view(1, 24, 1, 1, 1)
        std = torch.tensor(VIDEO_LATENTS_STD, device=device).view(1, 24, 1, 1, 1)
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, 3, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, 3, 1, 1, 1)
        for index in range(count):
            update(f"Recovering image {index + 1} of {count}…")
            raw = latent[:, :, index:index+1].to(device).float() * std + mean
            decoded = vae._decode_clip(raw.to(torch.float16))
            pixels = ((decoded.float() * pixel_std + pixel_mean).clamp(0, 1) * 255).byte()
            view_id = f"view-{index + 1:04}"
            Image.fromarray(pixels[0, :, -1].permute(1, 2, 0).cpu().numpy()).save(destination / f"{view_id}.png")
            items.append({"id": view_id, "latent_index": index})
            del raw, decoded, pixels
    return {"source": "refmod", "items": items, "notes": notes}


def preview_refmod(latent, destination: Path, kind: str, *, metadata=None, update=lambda _: None) -> dict:
    """Import with PNG masters plus the existing visual-language preview."""
    from .character_views import build_cache, make_preview

    views = build_cache(destination, lambda cache: recover_refmod_images(
        latent, cache, metadata or {"kind": kind}, update))
    return {"path": make_preview(destination, views, kind), "image_views": views}
