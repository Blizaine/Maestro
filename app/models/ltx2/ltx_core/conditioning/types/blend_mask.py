"""Latent-space blend mask for Sora-style clip transitions.

Blends two source video latents with a per-frame weight gradient AND a
per-frame denoise gradient that gives the model maximum creative freedom
in the middle of the transition while keeping the edges locked to each
clip's actual motion.

Latent blend:  A ━━━━━━━╲╱━━━━━━━ B   (A's latent ramps down, B's ramps up)
Denoise mask:  low ━━━ high ━━━ low   (edges: preserve motion, middle: creative)

At the edges the model is constrained to follow the source clip's motion
(the drone keeps rotating, the joggers keep their stride). In the middle
the model has freedom to generate a creative transition between the two
scenes. This is fundamentally different from a dissolve (uniform denoise)
or a single-frame morph (no source motion at all).
"""

import math
import torch

from ..item import ConditioningItem
from ...tools import LatentTools
from ...types import LatentState


class LatentBlendMask(ConditioningItem):
    """
    Conditioning that blends two source video latents with opposing weight
    gradients AND a bell-curve denoise mask.

    The denoise mask is:
      - LOW at frame 0 → model preserves Clip A's motion
      - HIGH at frame N/2 → model has creative freedom for the transition
      - LOW at frame N → model preserves Clip B's motion
    """

    def __init__(
        self,
        source_latent_a: torch.Tensor,
        source_latent_b: torch.Tensor,
        weight_a: float = 1.0,
        weight_b: float = 1.0,
        denoise_strength: float = 0.5,
    ):
        """
        Args:
            source_latent_a: [B, C, F, H, W] VAE-encoded clip A tail
            source_latent_b: [B, C, F, H, W] VAE-encoded clip B head (same shape)
            weight_a: Peak weight for clip A (applied at frame 0, ramps to 0)
            weight_b: Peak weight for clip B (applied at last frame, ramps from 0)
            denoise_strength: Controls the bell curve peak (0-1).
                0.5 = edges at ~0.15 denoise, middle at ~0.85
                0.7 = edges at ~0.10, middle at ~0.95 (more creative)
                0.3 = edges at ~0.20, middle at ~0.60 (more constrained)
        """
        self.source_latent_a = source_latent_a
        self.source_latent_b = source_latent_b
        self.weight_a = weight_a
        self.weight_b = weight_b
        self.denoise_strength = denoise_strength

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        tokens_a = latent_tools.patchifier.patchify(self.source_latent_a)
        tokens_b = latent_tools.patchifier.patchify(self.source_latent_b)

        num_tokens = min(tokens_a.shape[1], tokens_b.shape[1], latent_state.latent.shape[1])

        # Per-token weights for latent blending AND denoise mask
        blend_weights = torch.zeros(1, num_tokens, 1, device=tokens_a.device, dtype=tokens_a.dtype)
        denoise_mask_dim = latent_state.denoise_mask.shape[-1]  # could be 1 (per-token) or more
        denoise_mask = torch.zeros(1, num_tokens, denoise_mask_dim, device=tokens_a.device, dtype=tokens_a.dtype)

        # Bell curve with locked edges and moderate middle.
        #
        # Edges (mask ≈ 0.0): model preserves source clip motion exactly
        # Middle (mask ≈ 0.65): model sees the blended latent (knows what
        #   both scenes look like) but has enough freedom to generate real
        #   motion transitions instead of reproducing a dissolve.
        #
        # denoise_strength controls:
        #   0.3 = conservative middle (~0.45), wide edges (25% each)
        #   0.5 = balanced middle (~0.65), standard edges (15% each)
        #   0.7 = creative middle (~0.80), narrow edges (10% each)
        edge_fraction = max(0.05, 0.30 - self.denoise_strength * 0.30)
        # Middle peak: enough freedom for creative transitions but retains
        # visual hints from both clips so the model knows WHERE to go.
        peak_denoise = min(0.85, 0.35 + self.denoise_strength * 0.55)

        for t in range(num_tokens):
            frac = t / max(1, num_tokens - 1)  # 0.0 → 1.0

            # Latent blend: linear ramp from A to B
            alpha_a = (1.0 - frac) * self.weight_a
            alpha_b = frac * self.weight_b
            total = alpha_a + alpha_b
            blend_weights[0, t, 0] = (alpha_b / total) if total > 0 else 0.5

            # Denoise mask: smooth bell curve
            # sin²(π·t) gives 0 at edges, 1 at midpoint
            bell = math.sin(math.pi * frac) ** 2

            # Scale: 0 at edges → peak_denoise in the middle
            # But clamp the first/last edge_fraction to hard 0
            if frac < edge_fraction:
                ramp = frac / edge_fraction
                ramp = 0.5 - 0.5 * math.cos(math.pi * ramp)  # smooth
                mask_val = ramp * peak_denoise * bell / max(0.01, math.sin(math.pi * edge_fraction) ** 2)
                mask_val = min(mask_val, peak_denoise)
            elif frac > (1.0 - edge_fraction):
                ramp = (1.0 - frac) / edge_fraction
                ramp = 0.5 - 0.5 * math.cos(math.pi * ramp)
                tail_bell = math.sin(math.pi * (1.0 - edge_fraction)) ** 2
                mask_val = ramp * peak_denoise * bell / max(0.01, tail_bell)
                mask_val = min(mask_val, peak_denoise)
            else:
                mask_val = peak_denoise * bell

            denoise_mask[0, t, :] = mask_val

        # Blend the tokens: lerp from A to B
        blended = tokens_a[:, :num_tokens] * (1.0 - blend_weights) + tokens_b[:, :num_tokens] * blend_weights

        latent_state = latent_state.clone()

        # Set blended tokens as both current latent and clean reference
        latent_state.latent[:, :num_tokens] = blended
        latent_state.clean_latent[:, :num_tokens] = blended

        # Bell-curve denoise mask: edges preserved, middle free
        latent_state.denoise_mask[:, :num_tokens] = denoise_mask

        return latent_state
