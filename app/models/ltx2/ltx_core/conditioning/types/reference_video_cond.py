import torch

from ...components.patchifiers import get_pixel_coords
from ...tools import VideoLatentTools
from ...types import LatentState, VideoLatentShape
from ..item import ConditioningItem


class VideoConditionByReferenceLatent(ConditioningItem):
    def __init__(
        self,
        latent: torch.Tensor,
        strength: float = 1.0,
        frame_idx: int = 0,
        downscale_factor: int = 1,
        reference_cross_mask: torch.Tensor | None = None,
    ):
        self.latent = latent
        self.strength = strength
        self.frame_idx = int(frame_idx)
        self.downscale_factor = max(1, int(downscale_factor))
        self.reference_cross_mask = reference_cross_mask

    def apply_to(self, latent_state: LatentState, latent_tools: VideoLatentTools) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.latent)
        latent_shape = VideoLatentShape.from_torch_shape(self.latent.shape)
        positions = get_pixel_coords(
            latent_coords=latent_tools.patchifier.get_patch_grid_bounds(output_shape=latent_shape, device=self.latent.device),
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix if self.frame_idx == 0 else False,
        ).to(dtype=torch.float32)

        frame_idx = self.frame_idx
        remove_prepend = frame_idx < 0
        if remove_prepend:
            frame_idx = -frame_idx
        positions[:, 0, ...] += frame_idx
        positions[:, 0, ...] /= latent_tools.fps
        if self.downscale_factor != 1:
            positions[:, 1, ...] *= self.downscale_factor
            positions[:, 2, ...] *= self.downscale_factor

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.latent.device,
            dtype=self.latent.dtype,
        )
        cross_tokens = None
        if self.reference_cross_mask is not None:
            cross_mask = self.reference_cross_mask.to(
                device=self.latent.device,
                dtype=self.latent.dtype,
            )
            cross_tokens = latent_tools.patchifier.patchify(cross_mask)
            cross_tokens = cross_tokens.float().mean(dim=-1).clamp(0.0, 1.0)
        if remove_prepend:
            frame_tokens = latent_tools.patchifier.get_token_count(latent_shape._replace(frames=1))
            tokens = tokens[:, frame_tokens:]
            denoise_mask = denoise_mask[:, frame_tokens:]
            positions = positions[:, :, frame_tokens:]
            if cross_tokens is not None:
                cross_tokens = cross_tokens[:, frame_tokens:]

        old_token_count = latent_state.latent.shape[1]
        reference_token_count = tokens.shape[1]
        attention_mask = latent_state.attention_mask
        if cross_tokens is not None:
            if cross_tokens.shape[:2] != tokens.shape[:2]:
                raise ValueError(
                    "Reference cross-attention mask does not match the "
                    "encoded reference-token layout."
                )

            # Match Diffusers/LTX's IC-LoRA attention layout. The complete
            # reference canvas stays in the sequence so its positions define
            # the output canvas, while noisy/output tokens attend to each
            # reference token with the fractional strength produced by area
            # downsampling the pixel-space source mask.
            total_token_count = old_token_count + reference_token_count
            attention_mask = torch.zeros(
                (
                    tokens.shape[0],
                    total_token_count,
                    total_token_count,
                ),
                device=tokens.device,
                dtype=tokens.dtype,
            )
            if latent_state.attention_mask is not None:
                attention_mask[
                    :, :old_token_count, :old_token_count
                ] = latent_state.attention_mask.to(
                    device=tokens.device,
                    dtype=tokens.dtype,
                )
            else:
                attention_mask[
                    :, :old_token_count, :old_token_count
                ] = 1.0
            cross = cross_tokens.to(dtype=tokens.dtype)
            attention_mask[
                :, :old_token_count, old_token_count:
            ] = cross.unsqueeze(1)
            attention_mask[
                :, old_token_count:, :old_token_count
            ] = cross.unsqueeze(2)
            attention_mask[
                :, old_token_count:, old_token_count:
            ] = 1.0
            mask_mib = (
                attention_mask.numel()
                * attention_mask.element_size()
                / (1024.0 * 1024.0)
            )
            print(
                "[LTX2] Reference attention layout: "
                f"{old_token_count} output + {reference_token_count} "
                "reference tokens; source-reference weight "
                f"{float(cross.float().sum().item()):.1f}/"
                f"{cross.numel()} ({int((cross > 0).sum().item())} "
                f"tokens contributing); {mask_mib:.1f} MiB weighted mask."
            )
        elif attention_mask is not None:
            # Preserve an earlier conditioning mask if another fully visible
            # reference group is appended later.
            expanded_mask = torch.ones(
                (
                    tokens.shape[0],
                    old_token_count + reference_token_count,
                    old_token_count + reference_token_count,
                ),
                device=tokens.device,
                dtype=torch.bool,
            )
            expanded_mask[
                :, :old_token_count, :old_token_count
            ] = attention_mask > 0
            attention_mask = expanded_mask

        keyframes_mask = None
        if latent_state.keyframes_mask is not None:
            keyframes_mask = torch.cat(
                [latent_state.keyframes_mask, torch.zeros_like(denoise_mask)],
                dim=1,
            )

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=attention_mask,
            keyframes_mask=keyframes_mask,
            runtime_cache=latent_state.runtime_cache,
        )
