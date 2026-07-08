import math

import torch

from .protocols import DiffusionStepProtocol
from ..utils import to_velocity


class EulerDiffusionStep(DiffusionStepProtocol):
    """
    First-order Euler method for diffusion sampling (deterministic).
    Takes a single step from the current noise level (sigma) to the next by
    computing velocity from the denoised prediction and applying: sample + velocity * dt.
    """

    def step(
        self, sample: torch.Tensor, denoised_sample: torch.Tensor, sigmas: torch.Tensor, step_index: int
    ) -> torch.Tensor:
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        dt = sigma_next - sigma
        velocity = to_velocity(sample, sigma, denoised_sample)

        return (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)


class EulerAncestralDiffusionStep(DiffusionStepProtocol):
    """
    Euler ancestral sampling (stochastic).
    Like standard euler but adds controlled random noise after each step,
    producing more varied and natural-looking results at the cost of
    determinism. Matches ComfyUI's euler_ancestral_cfg_pp behavior at CFG=1.

    The noise injection is controlled by eta (default 1.0 = full ancestral).
    """

    def __init__(self, generator: torch.Generator | None = None, eta: float = 1.0):
        self._generator = generator
        self._eta = eta

    def step(
        self, sample: torch.Tensor, denoised_sample: torch.Tensor, sigmas: torch.Tensor, step_index: int
    ) -> torch.Tensor:
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]

        sigma_val = float(sigma.item()) if torch.is_tensor(sigma) else float(sigma)
        sigma_next_val = float(sigma_next.item()) if torch.is_tensor(sigma_next) else float(sigma_next)

        if sigma_next_val == 0:
            # Last step: no noise injection, just standard euler
            dt = sigma_next - sigma
            velocity = to_velocity(sample, sigma, denoised_sample)
            return (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)

        # Ancestral step: split sigma_next into deterministic (down) and stochastic (up) parts
        sigma_up = min(sigma_next_val, self._eta * math.sqrt(sigma_next_val ** 2 * (sigma_val ** 2 - sigma_next_val ** 2) / sigma_val ** 2))
        sigma_down = math.sqrt(sigma_next_val ** 2 - sigma_up ** 2)

        # Deterministic euler step toward sigma_down
        dt = sigma_down - sigma_val
        velocity = to_velocity(sample, sigma, denoised_sample)
        result = (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)

        # Stochastic noise injection
        if sigma_up > 0:
            noise = torch.randn(
                result.shape, dtype=result.dtype, device=result.device,
                generator=self._generator,
            )
            result = result + noise * sigma_up

        return result


class DPMSolverPlusPlus2MDiffusionStep(DiffusionStepProtocol):
    """
    DPM-Solver++ 2M — 2nd-order multistep deterministic sampler.

    Uses the PREVIOUS step's denoised prediction as a velocity-memory
    correction, achieving 2nd-order accuracy without requiring a midpoint
    model re-evaluation (the way Heun's / true RK2 would). One denoise call
    per step, same cost as Euler, but materially better sample quality —
    widely used in diffusers / ComfyUI (samplers: 'dpmpp_2m').

    Mathematically closer to ComfyUI's ClownSampler 'res_2s' in behavior
    than plain Euler, without the RES4LYF package dependency.

    First step (no previous denoised yet) and last step (sigma_next → 0,
    correction term unstable) fall back to Euler — identical math there.

    The stepper accumulates state across step() calls within a single
    denoising run; instantiate a fresh one per generation/stage.
    """

    def __init__(self):
        self._prev_denoised: torch.Tensor | None = None
        self._prev_sigma: float | None = None

    def step(
        self, sample: torch.Tensor, denoised_sample: torch.Tensor, sigmas: torch.Tensor, step_index: int
    ) -> torch.Tensor:
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        sigma_val = float(sigma.item()) if torch.is_tensor(sigma) else float(sigma)
        sigma_next_val = float(sigma_next.item()) if torch.is_tensor(sigma_next) else float(sigma_next)

        # Detect a "redundant" call where the stepper is invoked twice with the
        # SAME step_index (LTX-2's joint denoising loop in helpers.py:613-614
        # calls `stepper.step(video, ..., step_idx)` then
        # `stepper.step(audio, ..., step_idx)` — second call has prev_sigma ==
        # current sigma, so h_prev = log(σ/σ) = 0, r = 0, and the
        # `1/(2*r)` blend coefficient divides by zero.
        # Treat the redundant call as Euler (lower-order but stable) and
        # DON'T overwrite _prev_* so the next REAL step still has the right
        # prev_sigma from the previous step_idx. Side effect: audio runs
        # at Euler quality while video runs at DPM++ 2M; clean fix is to
        # use separate steppers in distilled.py (deferred follow-up).
        redundant_call = (
            self._prev_denoised is not None
            and self._prev_sigma is not None
            and self._prev_sigma == sigma_val
        )

        if redundant_call or self._prev_denoised is None or sigma_next_val == 0 or sigma_val <= 0:
            # First step or final step — fall back to Euler. For DPM++ 2M the
            # first iteration has no memory yet, and at σ→0 the correction
            # term is numerically unstable.
            dt = sigma_next - sigma
            velocity = to_velocity(sample, sigma, denoised_sample)
            result = (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)
        else:
            # 2nd-order multistep correction in log-σ parameterization.
            # t = -ln(σ), h = t_next - t = ln(σ / σ_next) ; h_prev similarly.
            # Ratio r = h_prev / h drives the blending coefficient.
            prev_sigma_val = self._prev_sigma
            h = math.log(sigma_val / max(sigma_next_val, 1e-8))
            h_prev = math.log(max(prev_sigma_val, 1e-8) / max(sigma_val, 1e-8))
            r = h_prev / h if h > 1e-8 else 1.0
            # Blend current and previous denoised predictions
            blend_curr = 1.0 + 1.0 / (2.0 * r)
            blend_prev = 1.0 / (2.0 * r)
            denoised_d = (
                blend_curr * denoised_sample.to(torch.float32)
                - blend_prev * self._prev_denoised.to(torch.float32)
            )
            # x_next = (σ_next / σ) * x + (1 - σ_next / σ) * denoised_d
            # Equivalent to k-diffusion's dpmpp_2m update:
            #   x = (σ_next/σ) * x - expm1(-h) * denoised_d
            sigma_ratio = sigma_next_val / sigma_val
            result = (
                sample.to(torch.float32) * sigma_ratio
                + denoised_d * (1.0 - sigma_ratio)
            ).to(sample.dtype)

        # Store for next iteration's correction term — but ONLY for the
        # primary call (skip on redundant audio call so next step_idx still
        # sees the correct prev_sigma).
        if not redundant_call:
            self._prev_denoised = denoised_sample.detach()
            self._prev_sigma = sigma_val

        return result
