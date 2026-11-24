# Adapted from the pinned WanGP v20.10 pi-H3 implementation.
# WanGP Community License 2.0; see LICENSE-WanGP-Community-2.0.txt.

from __future__ import annotations

"""pi-H3 schedules and LoRA recognition for MiniMax H3 FL2VA."""

from pathlib import PurePosixPath
import re


PI_H3_REVISIONS = {
    "repo_id": "DeepBeepMeep/MiniMax-H3",
    "revision": "da05e475e3266b30efbdc98befdf7888a069bbcc",
}
PI_H3_SIGMA_COUNTS = (2, 4)
PI_H3_RAW_MAX_TIME = {2: 0.93, 4: 0.96}
PI_H3_FINAL_STEP_SCALE = 0.6
PI_H3_AUDIO_TAIL_LATENTS = 5


def pi_h3_sigmas(steps: int, shift: float = 12.0) -> list[float]:
    """Return the trained video sigma grid (excluding the terminal clean point)."""
    import torch

    steps = int(steps)
    if steps not in PI_H3_RAW_MAX_TIME:
        raise ValueError("pi-H3 supports only its trained 2-step and 4-step schedules.")
    shift = float(shift)
    if shift <= 0:
        raise ValueError("pi-H3 flow shift must be positive.")
    first = PI_H3_RAW_MAX_TIME[steps]
    raw = torch.linspace(
        first,
        first * PI_H3_FINAL_STEP_SCALE / (steps - 1 + PI_H3_FINAL_STEP_SCALE),
        steps,
        dtype=torch.float64,
    )
    return (shift * raw / (1.0 + (shift - 1.0) * raw)).tolist()


def _map_pi_h3_audio_sigmas(
    video_sigmas,
    *,
    video_shift: float = 12.0,
    audio_shift: float = 3.0,
):
    """Map pi-H3's trained video grid onto H3's paired audio flow grid."""
    import torch

    video_sigmas = torch.as_tensor(video_sigmas, dtype=torch.float32)
    base_sigmas = video_sigmas / (
        float(video_shift) + video_sigmas * (1.0 - float(video_shift))
    )
    return float(audio_shift) * base_sigmas / (
        1.0 + (float(audio_shift) - 1.0) * base_sigmas
    )


def _scale_pi_h3_initial_noise(
    video_rows,
    audio_rows,
    layout,
    *,
    num_audio_latents: int,
    audio_channels: int,
    video_sigma,
    audio_sigma,
) -> None:
    """Scale generated Pi-H3 noise while leaving attached condition rows clean."""
    video_start = int(layout.num_condition_video_rows)
    video_stop = int(video_rows.shape[0]) - int(
        layout.num_target_condition_video_rows
    )
    if video_stop > video_start:
        video_rows[video_start:video_stop].mul_(video_sigma.to(video_rows))

    audio_start = int(layout.num_condition_audio_rows)
    conditioned_latents = int(layout.num_target_condition_audio_latents)
    if int(num_audio_latents) > conditioned_latents:
        audio_stop = audio_start + int(num_audio_latents) * int(audio_channels)
        target_audio = audio_rows[audio_start:audio_stop].view(
            int(audio_channels), int(num_audio_latents), -1
        )
        target_audio[:, conditioned_latents:].mul_(audio_sigma.to(audio_rows))


def pi_h3_adapter_steps(path: str) -> int | None:
    """Recognize the two official regular pi-H3 FL2VA LoRAs by basename."""
    basename = PurePosixPath(str(path or "").replace("\\", "/")).name.lower()
    match = re.fullmatch(r"minimax_h3_pi_(2|4)step_lora_bf16\.safetensors", basename)
    return int(match.group(1)) if match else None


def pi_h3_dwd_adapter_steps(path: str) -> int | None:
    """Recognize the two official pi-H3 DWD FL2VA LoRAs by basename."""
    basename = PurePosixPath(str(path or "").replace("\\", "/")).name.lower()
    match = re.fullmatch(
        r"minimax_h3_fl2va_pi_(2|4)step_dwd_lora_bf16\.safetensors", basename
    )
    return int(match.group(1)) if match else None


def is_pi_h3_accelerator(path: str) -> bool:
    return pi_h3_adapter_steps(path) is not None or pi_h3_dwd_adapter_steps(path) is not None


def pi_h3_lora_incompatibility(model_def: dict | None, path: str) -> str | None:
    """Return the model-family mismatch for a pi-H3 acceleration adapter."""
    steps = pi_h3_adapter_steps(path)
    dwd_steps = pi_h3_dwd_adapter_steps(path)
    if steps is None and dwd_steps is None:
        return None
    definition = model_def or {}
    if dwd_steps is not None:
        if int(definition.get("dwd", 0) or 0) != dwd_steps:
            return "DWD adapters require their matching dedicated pi-H3 DWD model"
        return None
    if definition.get("dwd"):
        return "regular pi-H3 adapters use the regular VAE and cannot be used on a DWD model"
    architecture = str(definition.get("architecture") or "")
    if architecture not in {"minimax_h3", "minimax_h3_full"}:
        return "pi-H3 adapters require a MiniMax H3 FL2VA Frames model"
    if any(
        definition.get(flag)
        for flag in (
            "vdn",
            "minimax_h3_fasth3",
            "minimax_h3_hyperflow_flash",
            "minimax_h3_pika_flex",
            "minimax_h3_fused_turbo",
            "minimax_h3_baked_turbo",
            "minimax_h3_singularity",
            "minimax_h3_dasiwa",
            "minimax_h3_ptd_refasymopd",
            "minimax_h3_audio_only",
        )
    ):
        return "pi-H3 uses a fixed FL2VA schedule and cannot be combined with this H3 variant"
    return None


__all__ = [
    "PI_H3_REVISIONS",
    "PI_H3_SIGMA_COUNTS",
    "is_pi_h3_accelerator",
    "pi_h3_adapter_steps",
    "pi_h3_dwd_adapter_steps",
    "pi_h3_lora_incompatibility",
    "pi_h3_sigmas",
]
