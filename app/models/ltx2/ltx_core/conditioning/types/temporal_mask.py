"""Temporal region mask conditioning for retake pipeline.

Sets denoise_mask based on a temporal region. Tokens inside the region get
denoise_mask=1 (fully denoised/regenerated). Tokens outside get denoise_mask=0
(preserved from source via clean_latent blending in post_process_latent).
"""

import torch

from ..item import ConditioningItem
from ...tools import LatentTools
from ...types import LatentState


class TemporalRegionMask(ConditioningItem):
    """
    Conditioning that marks a temporal region for regeneration while preserving the rest.

    The source video latent is set as both `latent` and `clean_latent` for all tokens.
    Then denoise_mask is set to 0 (preserve) everywhere, except the retake region
    where it's set to 1 (regenerate).

    During denoising:
    - GaussianNoiser only adds noise where denoise_mask=1 (retake region gets noised)
    - timesteps_from_mask() gives timestep=0 for frozen tokens (transformer knows they're clean)
    - post_process_latent() blends: denoised * mask + clean * (1 - mask) at every step
    """

    def __init__(
        self,
        source_latent: torch.Tensor,
        start_frame: int,
        end_frame: int,
    ):
        """
        Args:
            source_latent: [B, C, F, H, W] VAE-encoded source video in latent space
            start_frame: latent-space start frame index (inclusive) for retake region
            end_frame: latent-space end frame index (exclusive) for retake region
        """
        self.source_latent = source_latent
        self.start_frame = start_frame
        self.end_frame = end_frame

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        # Patchify source video to get token representation
        source_tokens = latent_tools.patchifier.patchify(self.source_latent)

        # Compute token indices for the retake region boundaries
        # Same pattern as VideoConditionByLatentIndex (latent_cond.py:33-36)
        start_token = latent_tools.patchifier.get_token_count(
            latent_tools.target_shape._replace(frames=self.start_frame)
        )
        end_token = latent_tools.patchifier.get_token_count(
            latent_tools.target_shape._replace(frames=self.end_frame)
        )
        total_tokens = source_tokens.shape[1]

        # Clamp to valid range
        start_token = max(0, min(start_token, total_tokens))
        end_token = max(start_token, min(end_token, total_tokens))

        latent_state = latent_state.clone()

        # Set ALL tokens to source video (both latent and clean_latent)
        num_tokens = min(source_tokens.shape[1], latent_state.latent.shape[1])
        latent_state.latent[:, :num_tokens] = source_tokens[:, :num_tokens]
        latent_state.clean_latent[:, :num_tokens] = source_tokens[:, :num_tokens]

        # Default: preserve everything (denoise_mask = 0)
        latent_state.denoise_mask[:, :num_tokens] = 0.0

        # Mark retake region for regeneration (denoise_mask = 1)
        if end_token > start_token:
            latent_state.denoise_mask[:, start_token:end_token] = 1.0

        return latent_state
