"""Flow-map Euler sampler adapted from the BFL FLUX 3 reference; see LICENSE."""

from collections.abc import Callable
from contextlib import nullcontext

import torch
from torch import Tensor

from .dit import DiT, Packed


DEFAULT_STEPS = 16
DEFAULT_ALPHA = 6.93


def timesteps(steps: int = DEFAULT_STEPS, alpha: float = DEFAULT_ALPHA) -> Tensor:
    """Return the reference flow-map schedule from one to zero."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    t = torch.linspace(1.0, 0.0, steps + 1)
    return alpha * t / (1.0 + (alpha - 1.0) * t)


@torch.inference_mode()
def denoise_flow_map(
    dit: DiT,
    packed: Packed,
    schedule: Tensor,
    on_step: Callable[[int], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
    negative_packed: Packed | None = None,
    guidance_scale: float = 1.0,
    loras_slists=None,
) -> Tensor:
    """Integrate one sample's packed latent tokens with the distilled flow map.

    ``on_step`` runs after each Euler update. ``check_cancel`` is called before
    and after each DiT evaluation so an interrupted generation does not wait
    for the rest of the schedule.
    """
    device_type = packed.data["image"].device.type
    use_bf16 = device_type == "cuda"
    if loras_slists is not None:
        from mmgp import offload
        from shared.utils.loras_mutipliers import update_loras_slists
        update_loras_slists(dit, loras_slists, len(schedule) - 1)
    for step in range(len(schedule) - 1):
        if check_cancel is not None:
            check_cancel()
        if loras_slists is not None:
            offload.set_step_no_for_lora(dit, step)
        t = schedule[step].item()
        dt = (schedule[step + 1] - schedule[step]).item()
        packed.timesteps["image"].fill_(t)
        target = schedule[step + 1].reshape(1).to(packed.data["image"].device) if getattr(dit, "flow_map", True) else None
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
        with autocast:
            velocity = dit(packed, timesteps_ref=target)
            if negative_packed is not None:
                negative_packed.data["image"] = packed.data["image"]
                negative_packed.timesteps["image"].fill_(t)
                unconditional = dit(negative_packed, timesteps_ref=None)
                velocity = unconditional + guidance_scale * (velocity - unconditional)
        if check_cancel is not None:
            check_cancel()
        packed.data["image"] = packed.data["image"] + dt * velocity
        if on_step is not None:
            on_step(step)
        if check_cancel is not None:
            check_cancel()
    return packed.data["image"]
