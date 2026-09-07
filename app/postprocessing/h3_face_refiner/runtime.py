"""Maestro adapter for WanGP's H3 crop-refinement recipe.

The detector, identity tracker and compositing are upstream; model ownership
and the Ref2VA call use Maestro's existing H3 implementation and GPU queue.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
import torch

ASSET_REVISION = "304d34f7751f8ba9ca0eb55d5d10044234cdbfe2"
ASSET_REPO = "DeepBeepMeep/MiniMax-H3"
ASSET_FOLDER = "buffalo_l"
DETECTOR = "face_yolov8m.pt"
IDENTITY_FILES = ("det_10g.onnx", "2d106det.onnx", "w600k_r50.onnx")
TURBO_FILE = "minimax_h3_lightx2v_ref2v_turbo_4step_alpha8_v0.1_bf16.safetensors"
DEFAULT_PROMPT = (
    "<Subject 1> is the single person in <Picture 1>. A stabilized close-up "
    "of this same person in the control video. Preserve their identity, facial "
    "proportions, expression, gaze, head pose, hair, clothing, motion, lighting "
    "and background. Restore natural, sharp, temporally consistent eyes, mouth, "
    "teeth, skin and fine facial detail. Preserve the original performance and "
    "lip movements. Do not introduce another person, head, object or scene."
)


def window_starts(frame_count, window_size=243, overlap=18):
    if frame_count < 1:
        raise ValueError("The source video contains no frames")
    if window_size < 22 or window_size > 345 or (window_size - 5) % 17:
        raise ValueError("Face refinement windows must use 5 + 17n frames, from 22 to 345")
    if not 0 <= overlap < window_size:
        raise ValueError("Face refinement overlap must be smaller than the window")
    if frame_count <= window_size:
        return (0,)
    count = math.ceil((frame_count - overlap) / (window_size - overlap))
    return tuple(int(n) for n in np.linspace(0, frame_count - window_size, count).round())


def refinement_sigmas(steps, denoising_strength, strength, video_shift=12.0, audio_shift=3.0):
    """Upstream's shortened, shifted schedule with uniform face strength.

    Unlike skipping initial evaluations, this performs every requested step
    and starts from source latents at the exact noise level seen by H3.
    """
    if not 0 < denoising_strength <= 1 or not 0 < strength <= 1:
        raise ValueError("Face refinement strength must be in (0, 1]")
    if int(steps) != steps or not 4 <= steps <= 8:
        raise ValueError("Face refinement requires 4–8 denoising steps")
    total = max(int(steps), int(steps / denoising_strength))
    base = torch.linspace(steps / total, 0.0, int(steps) + 1, dtype=torch.float32)
    video = (video_shift * base / (1 + (video_shift - 1) * base)) * strength
    audio = audio_shift * base / (1 + (audio_shift - 1) * base)
    return video, audio


def ensure_tracking_assets(progress=None):
    import wgp
    from shared.utils import files_locator as fl
    if progress:
        progress("Preparing face detection models", None, None)
    names = (DETECTOR, *IDENTITY_FILES)
    if any(not fl.locate_file(f"{ASSET_FOLDER}/{name}", error_if_none=False) for name in names):
        wgp.process_files_def(repoId=ASSET_REPO, revision=ASSET_REVISION,
            sourceFolderList=[ASSET_FOLDER], fileList=[list(names)])
    return fl.locate_file(f"{ASSET_FOLDER}/{DETECTOR}"), str(
        __import__("pathlib").Path(fl.locate_file(f"{ASSET_FOLDER}/{IDENTITY_FILES[0]}")).parent)


def ensure_refinement_adapter(progress=None):
    import wgp
    from shared.utils import files_locator as fl
    # process_files_def preserves the HF source subfolder inside the target.
    relative = f"minimax_h3/face_refiner/loras/{TURBO_FILE}"
    path = fl.locate_file(relative, error_if_none=False)
    if not path:
        if progress:
            progress("Preparing H3 four-step refinement adapter", None, None)
        wgp.process_files_def(repoId=ASSET_REPO, revision=ASSET_REVISION,
            sourceFolderList=["loras"], targetFolderList=["minimax_h3/face_refiner"], fileList=[[TURBO_FILE]])
        path = fl.locate_file(relative)
    return path


@contextmanager
def refinement_model(choice="auto", progress=None):
    import wgp
    from mmgp import offload
    from shared.utils import files_locator as fl

    fused_type = "minimax_h3_ref2va_fused_turbo"
    fused_filename = "minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors"
    use_fused = choice == "fused" or (choice == "auto" and bool(fl.locate_file(fused_filename, error_if_none=False)))
    model_type = fused_type if use_fused else "minimax_h3_ref2va"
    # Existing generation LoRAs and sparse-attention settings must not leak
    # into a separate restoration pass. One owner under the GPU queue.
    encoder = getattr(wgp.wan_model, "text_encoder_variant", None)
    if not encoder:
        from urllib.parse import urlparse
        definition = wgp.get_model_def(model_type)
        variants = definition.get("minimax_h3_text_encoder_variants", {})
        preferred = definition.get("minimax_h3_text_encoder_default", "nvfp4_awq")
        order = list(dict.fromkeys([preferred, *variants]))
        encoder = next((name for name in order if variants.get(name, {}).get("URLs") and all(
            fl.locate_file("minimax_h3/" + urlparse(url).path.rsplit("/", 1)[-1], error_if_none=False)
            for url in variants[name]["URLs"])), preferred)
    attention = offload.shared_state.get("_attention", "sdpa")
    wgp.release_model()
    offload.shared_state["_attention"] = "sdpa"
    try:
        adapter = ensure_refinement_adapter(progress) if not use_fused else None
        if progress:
            progress("Loading H3 Face Refiner", None, None)
        wgp.wan_model, wgp.offloadobj = wgp.load_models(model_type,
            override_profile=wgp.get_default_profile("video"),
            minimax_h3_text_encoder=encoder)
        model = wgp.wan_model
        model.transformer.cache = None
        if not use_fused:
            # LightX2V Ref2VA is a standard accelerator, not the fixed PDD
            # preset. Its reduced-noise four-step schedule is required here.
            model.validate_loras([adapter])
            offload.load_loras_into_model(model.transformer, [adapter], [1.0],
                activate_all_loras=True, pinnedLora=False, maxReservedLoras=0,
                preprocess_sd=wgp.get_loras_preprocessor(model.transformer, model_type),
                split_linear_modules_map=getattr(model.transformer, "split_linear_modules_map", None))
            errors = getattr(model.transformer, "_loras_errors", [])
            if errors:
                raise RuntimeError(f"Could not load the H3 face-refinement adapter: {errors}")
            model.finalize_loras()
        yield model, model_type
    finally:
        wgp.release_model()
        offload.shared_state["_attention"] = attention


def refine_window(model, crop, reference_path, *, fps, audio, audio_rate, options, seed, abort=None, progress=None):
    from services.media_flow import _check_abort
    count = int(crop.shape[1])
    padded = max(22, 5 + math.ceil((count - 5) / 17) * 17)
    if padded > count:
        crop = torch.cat([crop, crop[:, -1:].expand(-1, padded - count, -1, -1)], dim=1)
    _check_abort(abort)
    def step(index, *_args, **_kwargs):
        if abort and abort():
            model._interrupt = True
        if progress:
            progress("Refining facial detail", max(0, index + 1), options["steps"])
    try:
        result = model.generate(DEFAULT_PROMPT, input_frames=crop,
            minimax_h3_references=[{"type": "image", "path": str(reference_path), "role": "Person"}],
            input_waveform=audio, input_waveform_sample_rate=audio_rate,
            frame_num=padded, height=crop.shape[-2], width=crop.shape[-1], fps=fps,
            sampling_steps=options["steps"], seed=seed, callback=step,
            set_progress_status=lambda label: progress(label, None, None) if progress else None,
            _face_refinement={"strength": options["strength"], "denoising_strength": 0.45})
        _check_abort(abort)
        if result is None:
            raise RuntimeError("H3 face refinement returned no video")
        return result["x"][:, :count].detach().float().cpu().add_(1).mul_(127.5).round_().clamp_(0, 255).byte()
    finally:
        model._interrupt = False
