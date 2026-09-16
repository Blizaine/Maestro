"""Six-step soundtrack refinement adapted from Wan2GP v12.71.

The generated full-resolution video is immutable during this pass. Only the
existing audio is partly re-noised; no source media or LoRA is re-injected.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from mmgp import offload

from .packing import build_packed_sequence, build_row_timesteps
from .scheduler import MiniMaxH3Scheduler

STEPS = 6
DENOISING_STRENGTH = 0.5


def refinement_unavailable(*, audio_only=False, pdd=False, fused=False, source_audio=False):
    if audio_only:
        return "Audio refinement is a video soundtrack feature"
    if pdd:
        return "Audio refinement is unavailable with fixed 8-step PDD adapters"
    if fused:
        return "Audio refinement requires a checkpoint whose LoRAs can be disabled; fused Turbo is unsupported"
    if source_audio:
        return "Audio refinement is unavailable while a source soundtrack controls generation"
    return ""


@contextmanager
def without_loras(transformer):
    active = list(getattr(transformer, "_loras_active_adapters", ()))
    scaling = dict(getattr(transformer, "_loras_scaling", {}) or {})
    step = getattr(transformer, "_lora_step_no", 0)
    try:
        offload.activate_loras(transformer, [])
        yield
    finally:
        offload.activate_loras(transformer, active, [scaling[name] for name in active])
        offload.set_step_no_for_lora(transformer, step)


def refinement_sigmas(shift):
    base = torch.linspace(1.0, 0.0, round(STEPS / DENOISING_STRENGTH) + 1)[-(STEPS + 1):]
    return float(shift) * base / (1 + (float(shift) - 1) * base)


def refine_audio(model, video_rows, audio_rows, layout, prompt, latent_shape, audio_latents, seed,
                 callback=None, set_progress_status=None):
    if set_progress_status:
        set_progress_status("H3 Audio Refinement: 6 extra steps")
    # Remove reference/keyframe prefixes before rebuilding a clean layout.
    video = video_rows[layout.num_condition_video_rows:]
    audio = audio_rows[layout.num_condition_audio_rows:].clone()
    if model.omni_reference:
        context, tags = model.conditioner.forward_ref2va(prompt, model.device, [])
    else:
        context, tags = model.conditioner(prompt, model.device, None)
    if context is None or model._interrupt:
        return None
    frames, height, width = latent_shape
    refined_layout = build_packed_sequence(tags, frames, height, width, audio_latents, model.patch_size,
        (), target_condition_video_frames=frames)
    video_sigmas = refinement_sigmas(model.scheduler.shift)
    scheduler = MiniMaxH3Scheduler(shift=model.audio_scheduler.shift)
    scheduler.set_timesteps(sigmas=refinement_sigmas(model.audio_scheduler.shift), device=model.device)
    noise = torch.randn(audio.shape, generator=torch.Generator(device="cpu").manual_seed(seed),
                        dtype=torch.float32, device="cpu").to(audio)
    audio.lerp_(noise, scheduler.sigmas[0])
    del noise
    if callback:
        callback(-1, None, True, override_num_inference_steps=STEPS)
    with without_loras(model.transformer):
        for step, audio_timestep in enumerate(scheduler.timesteps):
            if model._interrupt:
                return None
            times, indices = build_row_timesteps(refined_layout, 1 - float(video_sigmas[step]),
                float(audio_timestep), 0.999, 1.0)
            prediction = model.transformer(hidden_states=video[None], audio_hidden_states=audio[None],
                encoder_hidden_states=context, timestep=times.to(model.device), timestep_indices=indices.to(model.device),
                token_tags=refined_layout.token_tags, position_ids=refined_layout.position_ids,
                video_indices=refined_layout.video_indices, audio_indices=refined_layout.audio_indices,
                text_indices=refined_layout.text_indices, return_dict=False, first_block_cache=None,
                packed_layout=refined_layout, latent_shape=latent_shape)
            if prediction is None or model._interrupt:
                return None
            # Discard the predicted video velocity. Never call the video
            # scheduler or assign into the original video storage.
            audio = scheduler.step(prediction[1][0].float(), audio_timestep, audio, return_dict=False)[0]
            if callback:
                callback(step, None)
    result = audio_rows.clone()
    result[layout.num_condition_audio_rows:] = audio
    return result
