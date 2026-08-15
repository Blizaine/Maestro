from dataclasses import dataclass, replace
from typing import Protocol

import torch
from torch._prims_common import DeviceLikeType

from .components.patchifiers import (
    AudioLatentShape,
    AudioPatchifier,
    VideoLatentPatchifier,
    VideoLatentShape,
    get_pixel_coords,
)
from .components.protocols import Patchifier
from .types import LatentState, SpatioTemporalScaleFactors

DEFAULT_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class LatentTools(Protocol):
    """
    Tools for building latent states.
    """

    patchifier: Patchifier
    target_shape: VideoLatentShape | AudioLatentShape

    def create_initial_state(
        self,
        device: DeviceLikeType,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        """
        Create an initial latent state. If initial_latent is provided, it will be used to create the latent state.
        """
        ...

    def patchify(self, latent_state: LatentState) -> LatentState:
        """
        Patchify the latent state.
        """
        if latent_state.latent.shape != self.target_shape.to_torch_shape():
            raise ValueError(
                f"Latent state has shape {latent_state.latent.shape}, expected shape is "
                f"{self.target_shape.to_torch_shape()}"
            )
        latent_state = latent_state.clone()
        latent = self.patchifier.patchify(latent_state.latent)
        clean_latent = self.patchifier.patchify(latent_state.clean_latent)
        denoise_mask = self.patchifier.patchify(latent_state.denoise_mask)
        keyframes_mask = (
            self.patchifier.patchify(latent_state.keyframes_mask)
            if latent_state.keyframes_mask is not None
            else None
        )
        return replace(
            latent_state,
            latent=latent,
            denoise_mask=denoise_mask,
            clean_latent=clean_latent,
            keyframes_mask=keyframes_mask,
        )

    def unpatchify(self, latent_state: LatentState) -> LatentState:
        """
        Unpatchify the latent state.
        """
        latent_state = latent_state.clone()
        latent = self.patchifier.unpatchify(latent_state.latent, output_shape=self.target_shape)
        clean_latent = self.patchifier.unpatchify(latent_state.clean_latent, output_shape=self.target_shape)
        denoise_mask = self.patchifier.unpatchify(
            latent_state.denoise_mask, output_shape=self.target_shape.mask_shape()
        )
        keyframes_mask = (
            self.patchifier.unpatchify(
                latent_state.keyframes_mask,
                output_shape=self.target_shape.mask_shape(),
            )
            if latent_state.keyframes_mask is not None
            else None
        )
        return replace(
            latent_state,
            latent=latent,
            denoise_mask=denoise_mask,
            clean_latent=clean_latent,
            keyframes_mask=keyframes_mask,
        )

    def clear_conditioning(self, latent_state: LatentState) -> LatentState:
        """
        Clear the conditioning from the latent state, returning ONLY the
        newly-generated tokens.

        Handles two cases:
          1. Conditioning items that APPEND tokens at the end (legacy
             video conditioning) — falls through with start_token=0.
          2. Conditioning items that PREPEND tokens at the start with
             negative temporal positions (Scenema's
             AudioConditionByReferenceLatent: prior chunk's tail as voice
             identity reference). Detects the prepended ref tokens via
             positions[..., 1] < 0 and slices past them.

        Ported from upstream Wan2GP. Maestro's original version sliced
        `latent[:, :num_tokens]` unconditionally — when ref tokens were
        prepended this returned the REFERENCE audio (e.g. chunk 1's
        tail) as if it were the new chunk's output, causing speakers
        to "repeat themselves from earlier lines" in multi-speaker
        dialogue.
        """
        num_tokens = self.patchifier.get_token_count(self.target_shape)
        start_token = 0
        positions = latent_state.positions
        if positions is not None and positions.ndim >= 4 and positions.shape[1] >= 1:
            ref_mask = positions[:, 0, :, 1] < 0
            if ref_mask.ndim == 2 and torch.any(ref_mask):
                counts = ref_mask.sum(dim=1)
                if torch.all(counts == counts[:1]):
                    ref_tokens = int(counts[0].item())
                    total_tokens = int(ref_mask.shape[1])
                    if 0 < ref_tokens < total_tokens and torch.all(ref_mask[:, :ref_tokens]) and not torch.any(ref_mask[:, ref_tokens:]):
                        start_token = ref_tokens

        stop_token = start_token + num_tokens
        latent = latent_state.latent[:, start_token:stop_token]
        clean_latent = latent_state.clean_latent[:, start_token:stop_token]
        denoise_mask = torch.ones_like(latent_state.denoise_mask[:, start_token:stop_token])
        positions = latent_state.positions[:, :, start_token:stop_token]

        keyframes_mask = (
            latent_state.keyframes_mask[:, start_token:stop_token]
            if latent_state.keyframes_mask is not None
            else None
        )

        # Conditioning attention is only meaningful while the appended
        # reference tokens are present. Dropping it here also avoids cloning
        # a potentially hundreds-of-MiB outpaint mask for every preview.
        return LatentState(
            latent=latent,
            denoise_mask=denoise_mask,
            positions=positions,
            clean_latent=clean_latent,
            keyframes_mask=keyframes_mask,
        )


@dataclass(frozen=True)
class VideoLatentTools(LatentTools):
    """
    Tools for building video latent states.
    """

    patchifier: VideoLatentPatchifier
    target_shape: VideoLatentShape
    fps: float
    scale_factors: SpatioTemporalScaleFactors = DEFAULT_SCALE_FACTORS
    causal_fix: bool = True

    def create_initial_state(
        self,
        device: DeviceLikeType,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        if initial_latent is not None:
            assert initial_latent.shape == self.target_shape.to_torch_shape(), (
                f"Latent shape {initial_latent.shape} does not match target shape {self.target_shape.to_torch_shape()}"
            )
            if initial_latent.device != device or initial_latent.dtype != dtype:
                initial_latent = initial_latent.to(device=device, dtype=dtype)
        else:
            initial_latent = torch.zeros(
                *self.target_shape.to_torch_shape(),
                device=device,
                dtype=dtype,
            )

        clean_latent = initial_latent.clone()

        denoise_mask = torch.ones(
            *self.target_shape.mask_shape().to_torch_shape(),
            device=device,
            dtype=torch.float32,
        )

        latent_coords = self.patchifier.get_patch_grid_bounds(
            output_shape=self.target_shape,
            device=device,
        )

        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self.scale_factors,
            causal_fix=self.causal_fix,
        ).float()
        positions[:, 0, ...] = positions[:, 0, ...] / self.fps

        state = self.patchify(
            LatentState(
                latent=initial_latent,
                denoise_mask=denoise_mask,
                # Per upstream WanGP commit c6b43fe ("the cure to ltx2 smearing"),
                # do NOT cast positions to the model dtype here — keep them as
                # the float32 produced by get_video_positions. Casting to bf16
                # caused subtle motion smearing in long-form LTX-2 outputs.
                positions=positions,
                clean_latent=clean_latent,
            )
        )
        keyframes_mask = torch.zeros_like(state.denoise_mask)
        first_frame_tokens = self.patchifier.get_token_count(
            self.target_shape._replace(frames=1)
        )
        keyframes_mask[:, :first_frame_tokens] = 1.0
        return replace(state, keyframes_mask=keyframes_mask)


@dataclass(frozen=True)
class AudioLatentTools(LatentTools):
    """
    Tools for building audio latent states.
    """

    patchifier: AudioPatchifier
    target_shape: AudioLatentShape

    def create_initial_state(
        self,
        device: DeviceLikeType,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        if initial_latent is not None:
            assert initial_latent.shape == self.target_shape.to_torch_shape(), (
                f"Latent shape {initial_latent.shape} does not match target shape {self.target_shape.to_torch_shape()}"
            )
            if initial_latent.device != device or initial_latent.dtype != dtype:
                initial_latent = initial_latent.to(device=device, dtype=dtype)
        else:
            initial_latent = torch.zeros(
                *self.target_shape.to_torch_shape(),
                device=device,
                dtype=dtype,
            )

        clean_latent = initial_latent.clone()

        denoise_mask = torch.ones(
            *self.target_shape.mask_shape().to_torch_shape(),
            device=device,
            dtype=torch.float32,
        )

        latent_coords = self.patchifier.get_patch_grid_bounds(
            output_shape=self.target_shape,
            device=device,
        )

        return self.patchify(
            LatentState(
                latent=initial_latent, denoise_mask=denoise_mask, positions=latent_coords, clean_latent=clean_latent
            )
        )

    def clear_conditioning(self, latent_state: LatentState) -> LatentState:
        """
        Strip conditioning tokens from the audio latent state.

        Standard parent impl slices `[:, :num_tokens]` from the front,
        assuming conditioning tokens are appended to the end. But
        AudioConditionByReferenceLatent PREPENDS reference tokens with
        negative temporal positions, so the front slice would keep the
        ref tokens and drop the actual target tokens.

        Ported from upstream WanGP v11.77 (ltx_core/tools.py): detect
        a contiguous prefix of tokens whose end-time is < 0 (the ref
        tokens) and start the slice AFTER them. Falls through to
        standard front-slice behavior when no ref tokens are detected.
        """
        num_tokens = self.patchifier.get_token_count(self.target_shape)
        start_token = 0
        positions = latent_state.positions
        if positions is not None and positions.ndim >= 4 and positions.shape[1] >= 1:
            ref_mask = positions[:, 0, :, 1] < 0
            if ref_mask.ndim == 2 and torch.any(ref_mask):
                counts = ref_mask.sum(dim=1)
                if torch.all(counts == counts[:1]):
                    ref_tokens = int(counts[0].item())
                    total_tokens = int(ref_mask.shape[1])
                    if 0 < ref_tokens < total_tokens and torch.all(ref_mask[:, :ref_tokens]) and not torch.any(ref_mask[:, ref_tokens:]):
                        start_token = ref_tokens

        stop_token = start_token + num_tokens
        latent = latent_state.latent[:, start_token:stop_token]
        clean_latent = latent_state.clean_latent[:, start_token:stop_token]
        denoise_mask = torch.ones_like(latent_state.denoise_mask[:, start_token:stop_token])
        positions = latent_state.positions[:, :, start_token:stop_token]

        return LatentState(
            latent=latent,
            denoise_mask=denoise_mask,
            positions=positions,
            clean_latent=clean_latent,
        )
