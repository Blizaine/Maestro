"""Spatial region mask conditioning for inpaint pipeline.

Sets denoise_mask based on a per-pixel spatial mask from SAM segmentation.
Masked pixels get denoise_mask=1 (regenerated), unmasked get denoise_mask=0 (preserved).
The mask is downsampled from pixel space to latent space before application.
"""

import torch
import torch.nn.functional as F

from ..item import ConditioningItem
from ...tools import LatentTools
from ...types import LatentState, SpatioTemporalScaleFactors


class SpatialRegionMask(ConditioningItem):
    """
    Conditioning that marks spatial regions for regeneration based on a SAM segmentation mask.

    The source video latent is set as both `latent` and `clean_latent`.
    The SAM mask (pixel space) is downsampled to latent space and applied to `denoise_mask`:
    - mask=1 (SAM detected region) → denoise_mask=1 → regenerate
    - mask=0 (background) → denoise_mask=0 → preserve from source
    """

    def __init__(
        self,
        source_latent: torch.Tensor,
        pixel_mask: torch.Tensor,
        scale_factors: SpatioTemporalScaleFactors | None = None,
    ):
        self.source_latent = source_latent
        self.pixel_mask = pixel_mask
        self.scale_factors = scale_factors or SpatioTemporalScaleFactors.default()

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        # Patchify source video to get tokens
        source_tokens = latent_tools.patchifier.patchify(self.source_latent)

        latent_state = latent_state.clone()

        # Set ALL tokens to source video (both latent and clean_latent)
        num_tokens = min(source_tokens.shape[1], latent_state.latent.shape[1])
        latent_state.latent[:, :num_tokens] = source_tokens[:, :num_tokens]
        latent_state.clean_latent[:, :num_tokens] = source_tokens[:, :num_tokens]

        # Default: preserve everything
        latent_state.denoise_mask[:] = 0.0

        # Build the mask in latent space [1, 1, F_lat, H_lat, W_lat]
        # matching the source_latent dimensions, then patchify it
        _, _, f_lat, h_lat, w_lat = self.source_latent.shape

        # Pixel mask: [T, H_px, W_px] → need to resize to [f_lat, h_lat, w_lat]
        mask = self.pixel_mask.float()  # [T, H_px, W_px]

        # Resize each frame of the mask to latent spatial dims
        # Use 2D interpolation per frame group for efficiency
        import numpy as np
        from PIL import Image

        mask_np = mask.cpu().numpy()  # [T, H_px, W_px]
        T_mask = mask_np.shape[0]

        # Create latent-space mask array
        latent_mask_np = np.zeros((f_lat, h_lat, w_lat), dtype=np.float32)

        # Map pixel frames to latent frames and downsample spatially
        time_scale = self.scale_factors[0]  # typically 8
        for lf in range(f_lat):
            # Which pixel frames contribute to this latent frame
            if lf == 0:
                pf_start, pf_end = 0, 1  # first latent = first pixel frame (causal)
            else:
                pf_start = (lf - 1) * time_scale + 1
                pf_end = min(lf * time_scale + 1, T_mask)

            pf_start = min(pf_start, T_mask - 1)
            pf_end = min(pf_end, T_mask)

            # Average the pixel masks in this temporal window
            if pf_end > pf_start:
                avg_mask = mask_np[pf_start:pf_end].mean(axis=0)  # [H_px, W_px]
            else:
                avg_mask = mask_np[pf_start]  # [H_px, W_px]

            # Resize spatially to latent dims using PIL (fast, handles any size)
            pil_mask = Image.fromarray((avg_mask * 255).astype(np.uint8))
            pil_resized = pil_mask.resize((w_lat, h_lat), Image.BILINEAR)
            latent_mask_np[lf] = (np.array(pil_resized) > 127).astype(np.float32)

        # Convert to tensor [1, 1, F_lat, H_lat, W_lat] and patchify
        latent_mask = torch.from_numpy(latent_mask_np).unsqueeze(0).unsqueeze(0)
        latent_mask = latent_mask.to(device=latent_state.denoise_mask.device, dtype=torch.float32)

        mask_tokens = latent_tools.patchifier.patchify(latent_mask)
        # mask_tokens: [1, num_tokens, patch_features] — for 1-channel mask, patch_features=1
        # Squeeze to match denoise_mask shape
        if mask_tokens.shape == latent_state.denoise_mask.shape:
            latent_state.denoise_mask[:, :num_tokens] = mask_tokens[:, :num_tokens]
        else:
            # Average across patch features if needed
            avg = mask_tokens.mean(dim=-1, keepdim=True)
            avg = (avg > 0.5).float()
            # Expand to match denoise_mask trailing dims
            while avg.ndim < latent_state.denoise_mask.ndim:
                avg = avg.unsqueeze(-1)
            nm = min(avg.shape[1], latent_state.denoise_mask.shape[1])
            latent_state.denoise_mask[:, :nm] = avg[:, :nm]

        # Debug: verify mask statistics
        total = num_tokens
        masked = (latent_state.denoise_mask[:, :num_tokens] > 0.5).sum().item()
        pix_masked = self.pixel_mask.sum().item() / self.pixel_mask.numel() * 100
        print(f"[SpatialMask] Pixel mask: {self.pixel_mask.shape} ({pix_masked:.1f}% masked)")
        print(f"[SpatialMask] Latent mask: [{f_lat}, {h_lat}, {w_lat}]")
        print(f"[SpatialMask] Applied: {masked}/{total} tokens masked ({masked/total*100:.1f}%)")

        return latent_state
