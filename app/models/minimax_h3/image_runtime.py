"""Native MiniMax H3 Image loading and packed image inference.

The transformer and row geometry follow the pinned Comfy-Org H3 Image
integration at commit ede4e5fe8d15445c6fcd6828d70b202c22f9502c. This keeps
the H3 Image audio-free six-head PDD checkpoint separate from the H3 Video
loader contract.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image

from shared.utils import files_locator as fl

from .packing import (
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PackedSequence,
    _spatial_position_grid,
    patchify_video_latents,
    unpatchify_video_tokens,
)
from .runtime import runtime


IMAGE_PDD_SIGMAS = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)
IMAGE_PDD_STEPS = len(IMAGE_PDD_SIGMAS) - 1
IMAGE_PATCH_SIZE = (1, 2, 2)
IMAGE_LATENT_CHANNELS = 24
IMAGE_OUTPUT_CHANNELS = IMAGE_LATENT_CHANNELS * 4
IMAGE_REFERENCE_NOISE_TIMESTEP = 0.999
IMAGE_MAX_REFERENCES = 16
IMAGE_REFERENCE_CANVAS_MULTIPLE = 32
IMAGE_REFERENCE_MAX_PIXELS = 4 * 1024 * 1024
IMAGE_MAX_PIXELS = 17_000_000


def _single_image_latent(value, *, name: str) -> torch.Tensor:
    """Normalize one already VAE-scaled image latent to ``[1,24,1,H,W]``."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor returned by the MiniMax H3 Image VAE.")
    if value.ndim == 4 and value.shape[0] == IMAGE_LATENT_CHANNELS:
        value = value.unsqueeze(0)
    if value.ndim != 5 or tuple(value.shape[:3]) != (1, IMAGE_LATENT_CHANNELS, 1):
        raise ValueError(
            f"{name} must have shape [1,24,1,H,W] (or [24,1,H,W]); "
            f"got {tuple(value.shape)}."
        )
    if value.shape[-2] % IMAGE_PATCH_SIZE[1] or value.shape[-1] % IMAGE_PATCH_SIZE[2]:
        raise ValueError(
            f"{name} height and width must be divisible by 2; "
            f"got {tuple(value.shape[-2:])}."
        )
    return value


def _image_spatial_grid(latent_height: int, latent_width: int) -> torch.Tensor:
    """Build one image's area-normalized 2x2 patch rotary grid."""

    sqrt_area = math.sqrt(latent_height * latent_width)
    # This is structural metadata, not model state. Keep it on CPU even when
    # MMGP has temporarily installed CUDA as PyTorch's default device.
    height = _spatial_position_grid(latent_height, IMAGE_PATCH_SIZE[1], sqrt_area).to(device="cpu")
    width = _spatial_position_grid(latent_width, IMAGE_PATCH_SIZE[2], sqrt_area).to(device="cpu")
    return torch.stack(
        [axis.reshape(-1) for axis in torch.meshgrid(height, width, indexing="ij")],
        dim=-1,
    ).to(device="cpu")


def build_image_packed_sequence(
    text_token_tags: torch.Tensor,
    target_latent_height: int,
    target_latent_width: int,
    reference_latents: Sequence[torch.Tensor] = (),
) -> MiniMaxH3PackedSequence:
    """Build the official ``[text | ordered references | target]`` H3 layout.

    Each reference gets its own normalized spatial grid and one rotary-time
    unit. The target starts after the final reference, exactly as the official
    ``PackedLayout`` image path does.
    """

    if target_latent_height <= 0 or target_latent_width <= 0:
        raise ValueError("MiniMax H3 Image target latent dimensions must be positive.")
    if target_latent_height % 2 or target_latent_width % 2:
        raise ValueError("MiniMax H3 Image target latent dimensions must be divisible by 2.")
    if len(reference_latents) > IMAGE_MAX_REFERENCES:
        raise ValueError(
            f"MiniMax H3 Image accepts at most {IMAGE_MAX_REFERENCES} reference images."
        )

    text_token_tags = torch.as_tensor(text_token_tags, dtype=torch.long, device="cpu").reshape(-1)
    text_length = int(text_token_tags.numel())
    target_rows = (target_latent_height // 2) * (target_latent_width // 2)
    ref_shapes = []
    ref_rows = []
    for index, value in enumerate(reference_latents):
        latent = _single_image_latent(value, name=f"reference_latents[{index}]")
        height, width = (int(dimension) for dimension in latent.shape[-2:])
        rows = (height // 2) * (width // 2)
        ref_shapes.append((height, width))
        ref_rows.append(rows)

    total_ref_rows = sum(ref_rows)
    sequence_length = text_length + total_ref_rows + target_rows
    position_ids = torch.zeros((sequence_length, 3), dtype=torch.float64, device="cpu")
    position_ids[:text_length, 0] = torch.arange(text_length, dtype=torch.float64, device="cpu")

    video_indices = []
    cursor = text_length
    reference_time = float(text_length)
    for (height, width), rows in zip(ref_shapes, ref_rows):
        grid = _image_spatial_grid(height, width)
        if grid.shape[0] != rows:
            raise RuntimeError("MiniMax H3 Image reference grid and packed row count disagree.")
        position_ids[cursor : cursor + rows, 0] = reference_time
        position_ids[cursor : cursor + rows, 1:] = grid
        video_indices.append(torch.arange(cursor, cursor + rows, dtype=torch.long, device="cpu"))
        cursor += rows
        reference_time += 1.0

    target_grid = _image_spatial_grid(target_latent_height, target_latent_width)
    if target_grid.shape[0] != target_rows:
        raise RuntimeError("MiniMax H3 Image target grid and packed row count disagree.")
    target_start = cursor
    position_ids[target_start : target_start + target_rows, 0] = reference_time
    position_ids[target_start : target_start + target_rows, 1:] = target_grid
    video_indices.append(torch.arange(target_start, target_start + target_rows, dtype=torch.long, device="cpu"))

    text_indices = torch.arange(text_length, dtype=torch.long, device="cpu")
    token_tags = torch.empty(sequence_length, dtype=torch.long, device="cpu")
    token_tags[text_indices] = text_token_tags
    video_index_tensor = torch.cat(video_indices)
    token_tags[video_index_tensor] = MINIMAX_H3_VIDEO_TAG

    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_index_tensor,
        audio_indices=torch.empty(0, dtype=torch.long, device="cpu"),
        text_indices=text_indices,
        num_condition_video_rows=total_ref_rows,
        num_condition_audio_rows=0,
    )


def _pdd_interval_head(
    packed_output: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    """Select one official H3 Image interval's base-plus-offset head."""

    if not 0 <= int(step_index) < IMAGE_PDD_STEPS:
        raise ValueError(
            f"MiniMax H3 Image step index must be in [0,{IMAGE_PDD_STEPS - 1}], "
            f"got {step_index}."
        )
    if packed_output.shape[-1] != IMAGE_OUTPUT_CHANNELS * IMAGE_PDD_STEPS:
        raise ValueError(
            "MiniMax H3 Image checkpoint must expose six stacked FP32 heads "
            f"({IMAGE_OUTPUT_CHANNELS * IMAGE_PDD_STEPS} output rows); "
            f"got {packed_output.shape[-1]}."
        )
    heads = packed_output.reshape(*packed_output.shape[:-1], IMAGE_PDD_STEPS, IMAGE_OUTPUT_CHANNELS)
    base = heads[..., 0, :]
    # Head 0 is the full prediction. Heads 1–5 are interval corrections. The
    # pinned six-knot sigma schedule spans exactly one bank interval per call.
    return base if step_index == 0 else base + heads[..., int(step_index), :]


def _patchify_reference_rows(latent: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply H3's 0.999 visual condition augmentation in packed row space."""

    rows = patchify_video_latents(latent.to(dtype=torch.float32), IMAGE_PATCH_SIZE)
    # Upstream restarts the CPU RNG from the request seed for each reference;
    # noise is sampled after patchification, so keep the row-space order here.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(rows.shape, generator=generator, dtype=torch.float32, device="cpu")
    return IMAGE_REFERENCE_NOISE_TIMESTEP * rows + (
        1.0 - IMAGE_REFERENCE_NOISE_TIMESTEP
    ) * noise.to(device=rows.device)


def _resize_reference_image(image: Image.Image, *, short_edge_min: int, short_edge_max: int) -> Image.Image:
    """Resize an ordered H3 reference proportionally on the 32-pixel canvas grid."""

    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("MiniMax H3 Image references must have nonzero dimensions.")
    short_edge = min(width, height)
    desired_short = min(max(short_edge, int(short_edge_min)), int(short_edge_max))
    scale = desired_short / short_edge
    if width * height * scale * scale > IMAGE_REFERENCE_MAX_PIXELS:
        scale = min(scale, math.sqrt(IMAGE_REFERENCE_MAX_PIXELS / (width * height)))

    def aligned(value: int) -> int:
        return max(
            IMAGE_REFERENCE_CANVAS_MULTIPLE,
            int(math.floor(value * scale / IMAGE_REFERENCE_CANVAS_MULTIPLE))
            * IMAGE_REFERENCE_CANVAS_MULTIPLE,
        )

    target = (aligned(width), aligned(height))
    if target == image.size:
        return image
    return image.resize(target, Image.Resampling.LANCZOS)


def _ordered_reference_images(value) -> list[Image.Image]:
    if value is None:
        return []
    images = list(value) if isinstance(value, (list, tuple)) else [value]
    if len(images) > IMAGE_MAX_REFERENCES:
        raise ValueError(
            f"MiniMax H3 Image accepts at most {IMAGE_MAX_REFERENCES} reference images."
        )
    if any(not isinstance(image, Image.Image) for image in images):
        raise ValueError("MiniMax H3 Image references must be PIL images.")
    return [image.convert("RGB") for image in images]


def _integer_setting(label: str, value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be an integer.") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be an integer.")
    return int(numeric)


class MiniMaxH3ImageRuntime:
    """Loaded image model, shared Qwen conditioner, and dedicated image VAE."""

    def __init__(self, transformer, conditioner, vae):
        self.transformer = transformer
        self.conditioner = conditioner
        self.text_encoder = conditioner.language_model
        self.vision_encoder = conditioner.visual
        self.vae = vae
        self.sigmas = IMAGE_PDD_SIGMAS
        self.__interrupt = False
        self._interrupt = False

    @property
    def _interrupt(self):
        return self.__interrupt

    @_interrupt.setter
    def _interrupt(self, value):
        self.__interrupt = bool(value)
        for module in (self.transformer, self.conditioner, self.text_encoder, self.vision_encoder, self.vae):
            if module is not None:
                try:
                    module._interrupt = self.__interrupt
                except (AttributeError, TypeError):
                    pass

    @torch.inference_mode()
    def encode_prompt(self, prompt: str, device, reference_images=()):
        """Return shared H3 Qwen embeddings and modality tags."""

        return self.conditioner(
            prompt,
            torch.device(device),
            images=list(reference_images) or None,
        )

    @torch.inference_mode()
    def predict_velocity(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_token_tags: torch.Tensor,
        reference_latents: Sequence[torch.Tensor] = (),
        *,
        sigma: float,
        step_index: int,
        reference_seed: int = 0,
    ) -> torch.Tensor | None:
        """Predict the data-ward H3 velocity for the target image latent.

        ``latents`` and reference latents are normalized MiniMax image-VAE
        latents. The return has the same shape as ``latents`` and is positive
        in H3's native convention: ``x0 = x_sigma + sigma * velocity``.
        """

        target = _single_image_latent(latents, name="latents")
        references = [
            _single_image_latent(value, name=f"reference_latents[{index}]")
            for index, value in enumerate(reference_latents)
        ]
        if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
            raise ValueError("MiniMax H3 Image prompt embeddings must have shape [1,L,5120].")
        if prompt_embeds.shape[-1] != 5120:
            raise ValueError(
                "MiniMax H3 Image requires the shared H3 Qwen layer-50 encoder "
                f"(5120 channels), got {prompt_embeds.shape[-1]}."
            )
        tags = torch.as_tensor(text_token_tags, dtype=torch.long, device="cpu").reshape(-1)
        if tags.numel() != prompt_embeds.shape[1]:
            raise ValueError(
                "MiniMax H3 Image prompt tag count must match encoded prompt length "
                f"({tags.numel()} != {prompt_embeds.shape[1]})."
            )
        if isinstance(sigma, bool) or not 0.0 <= float(sigma) <= 1.0:
            raise ValueError(f"MiniMax H3 Image sigma must be in [0,1], got {sigma!r}.")
        if self._interrupt or getattr(self.transformer, "_interrupt", False):
            return None

        device = target.device
        reference_rows = [
            _patchify_reference_rows(
                reference.to(device=device),
                reference_seed,
            )
            for reference in references
        ]
        target_rows = patchify_video_latents(
            target.to(device=device, dtype=torch.float32), IMAGE_PATCH_SIZE
        )
        hidden_states = torch.cat((*reference_rows, target_rows), dim=0).unsqueeze(0)
        layout = build_image_packed_sequence(
            tags,
            int(target.shape[-2]),
            int(target.shape[-1]),
            references,
        )
        if layout.video_indices.numel() != hidden_states.shape[1]:
            raise RuntimeError("MiniMax H3 Image packed media rows do not match latent rows.")

        t = 1.0 - float(sigma)
        ref_t = max(t, IMAGE_REFERENCE_NOISE_TIMESTEP)
        row_timesteps = torch.full(
            (layout.sequence_length,),
            t,
            dtype=torch.float32,
            device="cpu",
        )
        row_timesteps[layout.video_indices[: layout.num_condition_video_rows]] = ref_t
        timesteps, timestep_indices = torch.unique(
            row_timesteps,
            sorted=True,
            return_inverse=True,
        )
        audio_indices = torch.empty(0, dtype=torch.long, device="cpu")
        prediction = self.transformer(
            hidden_states=hidden_states,
            audio_hidden_states=None,
            encoder_hidden_states=prompt_embeds.to(device=device),
            timestep=timesteps.to(device=device),
            timestep_indices=timestep_indices.to(device=device),
            token_tags=layout.token_tags.to(device=device),
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=audio_indices,
            text_indices=layout.text_indices,
            return_dict=False,
            target_start_index=int(layout.text_indices.numel()) + layout.num_condition_video_rows,
            video_sink_tokens=int(layout.text_indices.numel()) + layout.num_condition_video_rows,
            condition_video_rows=layout.num_condition_video_rows,
        )
        if prediction is None or self._interrupt:
            return None
        video_output, audio_output = prediction
        if audio_output is not None:
            raise RuntimeError("The native MiniMax H3 Image transformer must remain audio-free.")
        target_row_count = target_rows.shape[0]
        target_output = video_output[:, -target_row_count:]
        velocity_rows = _pdd_interval_head(target_output, int(step_index))[0]
        return unpatchify_video_tokens(
            velocity_rows,
            num_latent_frames=1,
            latent_height=int(target.shape[-2]),
            latent_width=int(target.shape[-1]),
            channels=IMAGE_LATENT_CHANNELS,
            patch_size=IMAGE_PATCH_SIZE,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_prompt="",
        n_prompt="",
        seed=0,
        sampling_steps=IMAGE_PDD_STEPS,
        width=2048,
        height=2048,
        guide_scale=1.0,
        batch_size=1,
        input_ref_images=None,
        callback=None,
        set_progress_status=None,
        image_mode=1,
        video_prompt_type="",
        sample_solver="euler",
        **kwargs,
    ):
        """Generate one or more RGB images using H3's pinned six-step PDD path.

        WGP represents image batches in the frame axis, so the return value is
        ``[3, batch, height, width]`` RGB in ``[-1, 1]``.
        """

        # WGP reuses loaded model wrappers between jobs. Clear only the prior
        # request's interrupt here; the job lifecycle relay reasserts a durable
        # cancellation that races this reset.
        self._interrupt = False

        if str(image_mode).strip() not in ("1", "1.0"):
            raise ValueError("MiniMax H3 Image supports Image Generate mode only.")
        if str(video_prompt_type or "").strip() not in ("", "I", "KI"):
            raise ValueError("MiniMax H3 Image accepts ordered picture references only.")
        if str(sample_solver or "euler").strip().lower() not in ("euler", "default"):
            raise ValueError("MiniMax H3 Image requires its native six-step Euler sampler.")
        steps = _integer_setting("Sampling steps", sampling_steps)
        batch_size = _integer_setting("Batch size", batch_size)
        width = _integer_setting("Image width", width)
        height = _integer_setting("Image height", height)
        if steps != IMAGE_PDD_STEPS:
            raise ValueError(f"MiniMax H3 Image requires exactly {IMAGE_PDD_STEPS} sampling steps.")
        if batch_size <= 0:
            raise ValueError("MiniMax H3 Image batch size must be positive.")
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError("MiniMax H3 Image width and height must be positive multiples of 32.")
        if width * height > IMAGE_MAX_PIXELS:
            raise ValueError(
                f"MiniMax H3 Image supports up to {IMAGE_MAX_PIXELS:,} pixels per output image."
            )
        try:
            guidance = float(guide_scale)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("MiniMax H3 Image guidance must be exactly 1.") from error
        if not math.isfinite(guidance) or guidance != 1.0:
            raise ValueError("MiniMax H3 Image uses native single-pass conditioning with guidance 1.")
        if input_ref_images is None:
            for key in ("image_refs", "original_input_ref_images"):
                candidate = kwargs.get(key)
                if candidate is not None:
                    input_ref_images = candidate
                    break
        source_references = _ordered_reference_images(input_ref_images)
        if self._interrupt:
            return None

        def set_status(message: str) -> None:
            if callable(set_progress_status):
                set_progress_status(message)

        device = runtime.execution_device(self.transformer)
        vae_device = runtime.execution_device(self.vae)
        vision_references = [
            _resize_reference_image(image, short_edge_min=1024, short_edge_max=1024)
            for image in source_references
        ]
        vae_references = [
            _resize_reference_image(image, short_edge_min=1024, short_edge_max=2048)
            for image in source_references
        ]

        reference_latents = []
        if vae_references:
            set_status("Preparing Image Conditioning")
            for image in vae_references:
                if self._interrupt:
                    return None
                pixels = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
                pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(
                    device=vae_device,
                    dtype=torch.float32,
                )
                pixels = pixels.div_(127.5).sub_(1.0)
                try:
                    latent = self.vae.encode(pixels)
                except InterruptedError:
                    if not self._interrupt:
                        raise
                    return None
                if latent is None or self._interrupt:
                    return None
                reference_latents.append(
                    _single_image_latent(latent, name="encoded reference").to(
                        device=device,
                        dtype=torch.float32,
                    )
                )
                del pixels, latent

        set_status("Preparing Text Conditioning")
        prompt_embeds, text_token_tags = self.encode_prompt(
            str(input_prompt or ""),
            device,
            vision_references,
        )
        if prompt_embeds is None or text_token_tags is None or self._interrupt:
            return None

        total_steps = IMAGE_PDD_STEPS * batch_size
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=total_steps)

        set_status("Denoising")
        latent_height, latent_width = height // 16, width // 16
        generated_images = []
        base_seed = int(seed or 0)
        for image_index in range(batch_size):
            if self._interrupt:
                return None
            generator = torch.Generator(device="cpu").manual_seed(base_seed + image_index)
            latents = torch.randn(
                (1, IMAGE_LATENT_CHANNELS, 1, latent_height, latent_width),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(device=device)

            for step_index, (sigma, next_sigma) in enumerate(
                zip(IMAGE_PDD_SIGMAS[:-1], IMAGE_PDD_SIGMAS[1:])
            ):
                if self._interrupt:
                    return None
                velocity = self.predict_velocity(
                    latents,
                    prompt_embeds,
                    text_token_tags,
                    reference_latents,
                    sigma=sigma,
                    step_index=step_index,
                    reference_seed=base_seed,
                )
                if velocity is None or self._interrupt:
                    return None
                # H3 predicts data-ward velocity, x0 = x_sigma + sigma*v.
                # Euler integration over the descending sigma interval is
                # therefore a positive (sigma_i - sigma_{i+1}) step.
                latents = latents + (sigma - next_sigma) * velocity.to(
                    device=latents.device,
                    dtype=latents.dtype,
                )
                if callback is not None:
                    callback(image_index * IMAGE_PDD_STEPS + step_index, None, False)
                if self._interrupt:
                    return None

            if self._interrupt:
                return None
            set_status("VAE Decoding")
            try:
                pixels = self.vae.decode(latents.to(device=vae_device, dtype=torch.float32))
            except InterruptedError:
                if not self._interrupt:
                    raise
                return None
            if pixels is None or self._interrupt:
                return None
            if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
                raise RuntimeError(
                    "MiniMax H3 Image VAE must decode to RGB shaped [1,3,H,W], "
                    f"got {tuple(pixels.shape)}."
                )
            # WGP's image writer expects float samples in [-1, 1] and stores
            # the image batch on the sample's frame axis.
            generated_images.append(
                pixels[0].detach().to(device="cpu", dtype=torch.float32).mul(2.0).sub_(1.0).unsqueeze(1)
            )
            del latents, pixels

        if self._interrupt:
            return None
        return torch.cat(generated_images, dim=1)


def _resolve_checkpoint(path, *, description: str) -> str:
    if not path:
        raise FileNotFoundError(f"No {description} path was provided.")
    candidates = path if isinstance(path, (tuple, list)) else (path,)
    for candidate in candidates:
        resolved = fl.locate_file(str(candidate), error_if_none=False)
        if resolved:
            return resolved
    raise FileNotFoundError(f"Could not locate {description}: {path!r}.")


def _validate_image_schedule(path: str) -> None:
    """Check the image checkpoint's six-step PDD schedule before load."""

    from .turbo import safetensors_metadata

    raw_config = safetensors_metadata(path).get("config")
    try:
        config = json.loads(raw_config or "")
        transformer_config = config["transformer"]
        pdd_steps = int(transformer_config["pdd_steps"])
        sigmas = tuple(float(sigma) for sigma in transformer_config["pdd_sigmas"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "MiniMax H3 Image checkpoint metadata must declare its pdd_steps and pdd_sigmas."
        ) from error
    if pdd_steps != IMAGE_PDD_STEPS or sigmas != IMAGE_PDD_SIGMAS:
        raise ValueError(
            "MiniMax H3 Image checkpoint PDD schedule does not match the pinned six-step grid."
        )


def model_factory(
    model_filename,
    *,
    model_def: dict,
    text_encoder_filename=None,
    VAE_dtype: torch.dtype | None = None,
    dtype: torch.dtype | None = None,
    **_kwargs,
):
    """Load native H3 Image components and return ``(runtime, MMGP map)``."""

    from .image_vae import load_image_vae
    from .minimax_h3_handler import _TEXT_ENCODER
    from .minimax_h3_main import _load_conditioner, _load_transformer

    dtype = dtype or torch.bfloat16
    VAE_dtype = VAE_dtype or torch.float16
    assets_root = str(
        (model_def or {}).get("minimax_h3_assets_root")
        or (model_def or {}).get("text_encoder_folder")
        or "minimax_h3"
    )
    if text_encoder_filename is None:
        text_encoder_filename = os.path.join(assets_root, _TEXT_ENCODER)
    text_encoder_filename = _resolve_checkpoint(
        text_encoder_filename,
        description="MiniMax H3 32B text encoder",
    )
    if isinstance(model_filename, (tuple, list)):
        if len(model_filename) != 1:
            raise ValueError("MiniMax H3 Image expects one native transformer checkpoint file.")
        model_filename = model_filename[0]
    model_filename = _resolve_checkpoint(
        model_filename,
        description="MiniMax H3 Image transformer",
    )
    _validate_image_schedule(model_filename)
    transformer = _load_transformer(
        model_filename,
        dtype,
        qkv_layout="grouped",
        image_model=True,
    )
    conditioner = _load_conditioner(
        text_encoder_filename,
        assets_root,
        dtype,
        variant="nvfp4_awq",
    )
    vae_filename = _resolve_checkpoint(
        (model_def or {}).get("minimax_h3_image_vae_filename"),
        description="MiniMax H3 Image VAE",
    )
    vae = load_image_vae(vae_filename, dtype=VAE_dtype)
    runtime = MiniMaxH3ImageRuntime(transformer, conditioner, vae)
    pipe = {
        "transformer": runtime.transformer,
        "text_encoder": runtime.text_encoder,
        "vision_encoder": runtime.vision_encoder,
        "vae": runtime.vae,
    }
    return runtime, pipe


__all__ = [
    "IMAGE_PDD_SIGMAS",
    "IMAGE_PDD_STEPS",
    "MiniMaxH3ImageRuntime",
    "build_image_packed_sequence",
    "model_factory",
]
