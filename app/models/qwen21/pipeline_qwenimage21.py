# Vendored from Diffusers 80c7ed262aeffbeb43ef13ae04baeb9b84515a69; modified for Maestro. See THIRD_PARTY_NOTICES.md.
# Copyright 2026 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from .autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
from .transformer_qwenimage21 import QwenImage21Transformer2DModel
from .transformer_qwenimage21 import QwenImage21KVCache
from .memory import GIB, reference_cache_plan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# These are the published Viggle Turbo schedules used by Wan2GP. Keep the
# accelerated paths explicit: using a few steps with the base schedule does
# not reproduce the distilled adapter's training trajectory.
VIGGLE_SIGMAS = {
    4: (1.0, 0.75, 0.5, 0.25),
    5: (1.0, 0.875, 0.75, 0.5, 0.25),
    6: (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25),
}
_VIGGLE_PROFILE_STEPS = {"viggle_v01": 4, "viggle_v02": 5, "viggle_v021": 6}


def qwen21_sampling_profile(sample_solver: str, num_inference_steps: int):
    """Return scheduler overrides and sigma nodes for a named Qwen profile."""
    solver = (sample_solver or "default").lower()
    if solver in ("", "default", "euler"):
        return {}, None
    if solver in _VIGGLE_PROFILE_STEPS:
        required = _VIGGLE_PROFILE_STEPS[solver]
        if int(num_inference_steps) != required:
            raise ValueError(f"{solver} requires exactly {required} inference steps.")
        return {"shift_terminal": None}, VIGGLE_SIGMAS[int(num_inference_steps)]
    raise ValueError(f"Unknown Qwen Image 2.1 sampler profile: {sample_solver!r}.")


def _prepare_red_outpainting(images, width, height, outpainting_dims, input_masks, prompt):
    """Crop WanGP's padded control image and make the reference red-canvas prompt."""
    if outpainting_dims is None or not any(outpainting_dims):
        return images, input_masks, prompt, None, None, None
    if not images:
        raise ValueError("Outpainting requires a control image or a main reference image.")
    if images[0].size != (width, height):
        raise ValueError("The outpainting source must match the prepared canvas dimensions.")

    from shared.utils.utils import get_outpainting_frame_location

    source_height, source_width, top, left = get_outpainting_frame_location(
        height, width, outpainting_dims, 1, quantize_margins=32
    )
    location = (source_height, source_width, top, left)
    if any(value % 32 for value in location):
        raise ValueError("Outpainting source dimensions and offsets must align to 32 pixels.")
    if min(source_height, source_width) < 32:
        raise ValueError("Outpainting must retain at least a 32-pixel source in each dimension.")

    source = images[0].crop((left, top, left + source_width, top + source_height))
    canvas = PILImage.new("RGBA", (width, height), (255, 0, 0, 255))
    canvas.paste(source.convert("RGBA"), (left, top))
    from shared.utils.utils import convert_tensor_to_image

    if input_masks is None:
        mask_array = np.zeros((height, width), dtype=np.uint8)
    else:
        mask = convert_tensor_to_image(input_masks, mask_levels=True).convert("L")
        if mask.size != (width, height):
            mask = mask.resize((width, height), PILImage.Resampling.LANCZOS)
        mask_array = np.array(mask, copy=True)
    source_edit_mask = mask_array.copy()
    source_edit_mask[:top] = 0
    source_edit_mask[top + source_height:] = 0
    source_edit_mask[:, :left] = 0
    source_edit_mask[:, left + source_width:] = 0
    source_edit_mask = PILImage.fromarray(source_edit_mask) if np.any(source_edit_mask > 64) else None
    # White pixels are the area Qwen is allowed to regenerate. Outpainting
    # margins always run the full schedule, while source pixels stay anchored.
    mask_array[:top] = 255
    mask_array[top + source_height:] = 255
    mask_array[:, :left] = 255
    mask_array[:, left + source_width:] = 255
    input_masks = PILImage.fromarray(mask_array)
    suffix = "Remove the red paddings on the sides and show what's behind them."
    prompt = prompt or ""
    if suffix not in prompt:
        prompt = prompt.rstrip().rstrip(".") + ". " + suffix
    return [source, *images[1:]], input_masks, prompt, location, canvas, source_edit_mask


def _outpainting_overlap(location, width, height):
    """Return source-edge release bands, aligned to the model's 32px canvas grid."""
    source_height, source_width, top, left = location

    def bands(size, before, after):
        count = int(before) + int(after)
        desired = min(256, max(128, (size // 4 // 32) * 32))
        available = ((size - 32) // max(count, 1) // 32) * 32
        band = min(desired, available)
        return (band if before else 0), (band if after else 0)

    upper, lower = bands(source_height, top > 0, top + source_height < height)
    leading, trailing = bands(source_width, left > 0, left + source_width < width)
    return upper, lower, leading, trailing


def _outpainting_step_mask(location, width, height, sigma, device, dtype):
    """Generation weight for upstream-style gradual anchoring and edge regeneration."""
    source_height, source_width, top, left = location
    upper, lower, leading, trailing = _outpainting_overlap(location, width, height)
    mask = torch.ones((1, height // 16, width // 16), device=device, dtype=dtype)
    region = mask[:, top // 16:(top + source_height) // 16,
                   left // 16:(left + source_width) // 16]
    region[:, upper // 16:(source_height - lower) // 16,
           leading // 16:(source_width - trailing) // 16] = 0
    region.mul_((1 - sigma).pow(4))
    if upper:
        region[:, :2, :] = 1
    if lower:
        region[:, -2:, :] = 1
    if leading:
        region[:, :, :2] = 1
    if trailing:
        region[:, :, -2:] = 1
    return mask.reshape(1, -1, 1)


def _mask_to_latents(mask, width, height, device, dtype):
    if mask is None:
        return None
    from shared.utils.utils import convert_tensor_to_image

    if not isinstance(mask, PILImage.Image):
        mask = convert_tensor_to_image(mask, mask_levels=True)
    mask = mask.convert("L").resize((width // 16, height // 16), PILImage.Resampling.LANCZOS)
    values = torch.from_numpy(np.array(mask, copy=True)).to(device=device)
    return values.gt(64).to(dtype).reshape(1, -1, 1)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import QwenImage21Pipeline

        >>> pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A capybara wearing a wizard hat, reading a book by candlelight, oil painting"
        >>> image = pipe(prompt).images[0]
        >>> image.save("qwenimage21.png")
        ```
"""


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.flux.pipeline_flux_control_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.calculate_dimensions
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


class QwenImage21Pipeline(DiffusionPipeline, QwenImageLoraLoaderMixin):
    r"""
    Text-to-image and image-conditioned generation with Qwen-Image 2.1.

    Prompt and condition images are encoded together by a Qwen3-VL model, so a condition image occupies the vision
    slots the encoder reserved for it and the transformer sees one interleaved text/image sequence.

    Args:
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Scheduler used to denoise the encoded image latents.
        vae ([`AutoencoderKLQwenImage21`]):
            Variational auto-encoder mapping images to and from the 64-channel latent space.
        text_encoder ([`Qwen3VLForConditionalGeneration`]):
            Qwen3-VL model producing the joint text/image embeddings.
        processor ([`Qwen3VLProcessor`]):
            Processor that builds the chat template and tokenizes prompt and condition images.
        transformer ([`QwenImage21Transformer2DModel`]):
            The single-stream block-causal transformer that denoises the latents.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage21,
        text_encoder: Qwen3VLForConditionalGeneration,
        processor: Qwen3VLProcessor,
        transformer: QwenImage21Transformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
        )
        # The VAE compresses 16x spatially and the transformer consumes latents unpatched, so one token covers a 16x16
        # pixel tile.
        self.vae_scale_factor = 16
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 64
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, vae_latent_channels=self.latent_channels
        )
        self.sys_prompt = "Comprehend and analyze the provided prompt."
        # The prompt is built as a raw template string and passed straight to
        # `self.processor(text=..., images=...)`, rather than going through `apply_chat_template`:
        # the two tokenize differently and the checkpoint expects this one. The "Picture 1: ..."
        # vision prefix only appears in the image-conditioned template.
        self.prompt_template_t2i = (
            f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{{}}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        self.prompt_template_ti2i = (
            f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        # Number of leading system-role tokens to drop from the hidden states. Derived from the
        # tokenized system message rather than hardcoded, so it tracks the processor's template.
        sys_message = [{"role": "system", "content": [{"type": "text", "text": self.sys_prompt}]}]
        sys_tokens = self.processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
        self._drop_idx = len(sys_tokens[0] if isinstance(sys_tokens[0], (list, tuple)) else sys_tokens)
        self._img_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        image: list | None = None,
        device: torch.device | None = None,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # Qwen has no bos token, so an empty string leaves the encoder with nothing to read.
        prompt = [" " if not p else p for p in prompt]
        is_t2i = image is None

        if is_t2i:
            prompts = [self.prompt_template_t2i.format(t) for t in prompt]
        else:
            prompts = []
            condition_pil_list = []
            for t in prompt:
                n_imgs = len(image)
                replace = "<image1><|vision_start|><|image_pad|><|vision_end|>"
                for i in range(2, n_imgs + 1):
                    replace += f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
                template = self.prompt_template_ti2i.replace(
                    "<image1><|vision_start|><|image_pad|><|vision_end|>", replace
                )
                prompts.append(template.format(t))
            # Each prompt's template repeats the `<|image_pad|>` placeholders, so hand the processor one set of
            # images per prompt, in the order the placeholders appear.
            for _ in prompt:
                for img in image:
                    if not isinstance(img, PILImage.Image):
                        img = PILImage.fromarray(img)
                    if img.mode == "RGBA":
                        # The checkpoint was trained with the alpha composited over white for the vision encoder.
                        # Only this copy is flattened; the VAE still reads all four channels.
                        white = PILImage.new("RGB", img.size, (255, 255, 255))
                        white.paste(img, mask=img.getchannel("A"))
                        img = white
                    condition_pil_list.append(img)

        # Left padding, as the checkpoint was trained with. `_extract_masked_hidden` drops the padding either way,
        # but the side decides the positions the encoder sees for a batch of prompts of different lengths.
        processor_kwargs = {
            "text": prompts,
            "padding": True,
            "padding_side": "left",
            "return_tensors": "pt",
        }
        if not is_t2i:
            processor_kwargs["images"] = condition_pil_list

        model_inputs = self.processor(**processor_kwargs).to(device)

        forward_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": False,
            "use_cache": False,
        }
        if not is_t2i and hasattr(model_inputs, "pixel_values"):
            forward_kwargs.update(pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw)
        if hasattr(model_inputs, "mm_token_type_ids"):
            forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

        # `hidden_states[-1]` has to be the last decoder layer's output, before the text encoder's final RMSNorm:
        # that is what the transformer was trained on. It is what transformers 4.x returns there, but from
        # transformers 5.0 the output capturing ties that entry to `last_hidden_state`, so it comes back normalized
        # instead — a third of the signal the transformer reads, which shows up first in rendered text. A forward hook
        # returning the module's input replaces its output, which neutralizes the norm for this call on either version.
        # TODO: replace this with `tie_last_hidden_states=False` in the text encoder's config, which
        # huggingface/transformers#48087 adds, once that ships in a stable transformers release (5.18).
        encoder_model = getattr(self.text_encoder, "model", self.text_encoder)
        text_model = getattr(encoder_model, "language_model", encoder_model)
        # Maestro loads the backbone without its unused vocabulary head. Save
        # just this final pre-norm activation, rather than all 36 layer outputs.
        captured = []

        def capture_pre_norm(module, args, output):
            captured.append(args[0])
            return args[0]

        handle = text_model.norm.register_forward_hook(capture_pre_norm)
        try:
            self.text_encoder(**forward_kwargs)
        finally:
            handle.remove()
        hidden_states = captured.pop()

        split_hidden_states = list(self._extract_masked_hidden(hidden_states, model_inputs.attention_mask))
        split_hidden_states = [e[self._drop_idx :] for e in split_hidden_states]

        image_pad_mask = [
            (sample_ids[sample_mask.bool()] == self._img_token_id)
            for sample_ids, sample_mask in zip(model_inputs.input_ids, model_inputs.attention_mask)
        ]
        image_pad_mask = [e[self._drop_idx :] for e in image_pad_mask]

        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max(e.size(0) for e in split_hidden_states)
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        image_pad_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in image_pad_mask])

        return prompt_embeds, encoder_attention_mask, image_pad_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: list[PipelineImageInput] | None = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        image_pad_mask: torch.Tensor | None = None,
    ):
        r"""
        Args:
            prompt (`str` or `list[str]`, *optional*):
                Prompt to be encoded.
            image (`list[PipelineImageInput]`, *optional*):
                Condition images to encode alongside the prompt.
            device (`torch.device`):
                Torch device.
            num_images_per_prompt (`int`):
                Number of images generated per prompt.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Skips encoding when provided.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask, image_pad_mask = self._get_qwen_prompt_embeds(prompt, image, device)
        elif image_pad_mask is None:
            if image is not None:
                raise ValueError(
                    "Pass `image_pad_mask` alongside `prompt_embeds` when the embeddings cover condition images, so "
                    "the transformer knows which positions hold image tokens."
                )
            # Embeddings supplied without a mask can only be text, so no position holds an image token.
            image_pad_mask = prompt_embeds.new_zeros(prompt_embeds.shape[:2], dtype=torch.bool)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        # `repeat(1, n)` on the 2D mask, so its rows interleave the same way the 3D embeddings' do. With
        # `repeat(1, n, 1)` the mask picks up a leading axis and the rows come out tiled instead, which pairs each
        # sample with another prompt's padding.
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt)
            prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        # Without padding there is nothing to mask, and a mask that carries no information costs the attention
        # backends that reject one outright.
        if prompt_embeds_mask is not None and prompt_embeds_mask.all():
            prompt_embeds_mask = None

        return prompt_embeds, prompt_embeds_mask, image_pad_mask

    def check_inputs(self, prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and "
                f"{width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found "
                f"{[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Pass either `prompt` or `prompt_embeds`, not both.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Pass one of `prompt` or `prompt_embeds`.")

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        # 2.1 consumes latents unpatched, so packing is a plain spatial flatten.
        return latents.view(batch_size, num_channels_latents, height * width).transpose(1, 2)

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, _, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.transpose(1, 2).reshape(batch_size, channels, 1, height, width)
        return latents

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit.QwenImageEditPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        image = image.to(dtype=self.vae.dtype)
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    def prepare_latents(
        self, images, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        image_latents = None
        if images is not None:
            all_image_latents = []
            for image in images:
                image = image.to(device=device, dtype=dtype)
                encoded = self._encode_vae_image(image, generator).to(dtype=dtype)
                if batch_size > encoded.shape[0]:
                    if batch_size % encoded.shape[0] != 0:
                        raise ValueError(
                            f"Cannot duplicate `image` of batch size {encoded.shape[0]} to {batch_size} text prompts."
                        )
                    encoded = torch.cat([encoded] * (batch_size // encoded.shape[0]), dim=0)
                image_latent_height, image_latent_width = encoded.shape[3:]
                all_image_latents.append(
                    self._pack_latents(
                        encoded, batch_size, num_channels_latents, image_latent_height, image_latent_width
                    )
                )
            image_latents = torch.cat(all_image_latents, dim=1)

        if latents is None:
            shape = (batch_size, 1, num_channels_latents, height, width)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] = None,
        image: PipelineImageInput | None = None,
        negative_prompt: str | list[str] = None,
        true_cfg_scale: float = 1.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 40,
        sigmas: list[float] | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        output_resolution: int = 1024,
        use_kv_cache: bool = True,
        sample_solver: str = "default",
        mask_image: Any | None = None,
        inpaint_image: PipelineImageInput | None = None,
        denoising_strength: float = 1.0,
        masking_strength: float = 1.0,
        model_mode: int = 0,
        outpainting_dims: tuple[int, int, int, int] | None = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt to guide image generation. Pass `prompt_embeds` instead to supply embeddings directly.
            image (`PipelineImageInput`, *optional*):
                One or more condition images, as a PIL image or a numpy array. They are encoded by the text encoder as
                vision context and by the VAE into latent tokens prepended to the noise. A list is one set of images
                shared by every prompt in the batch, not one entry per prompt.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt not to guide image generation. Ignored when `true_cfg_scale` is not greater than 1.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                Classifier-free guidance scale. Enabled by `true_cfg_scale > 1` together with a negative prompt.
                Qwen-Image 2.1 is meant to be sampled without guidance, hence the default of 1.0.
            height (`int`, *optional*):
                Height in pixels of the generated image. Derived from the condition image's aspect ratio if omitted.
            width (`int`, *optional*):
                Width in pixels of the generated image. Derived from the condition image's aspect ratio if omitted.
            num_inference_steps (`int`, *optional*, defaults to 40):
                Number of denoising steps.
            sigmas (`list[float]`, *optional*):
                Custom sigmas for the denoising schedule.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images generated per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Generator(s) to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings, which skip prompt encoding. Pass `prompt_embeds_mask` with them.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Bool mask marking the valid positions of `prompt_embeds`.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings, used in place of `negative_prompt`. Pass
                `negative_prompt_embeds_mask` with them.
            negative_prompt_embeds_mask (`torch.Tensor`, *optional*):
                Bool mask marking the valid positions of `negative_prompt_embeds`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format, `"pil"`, `"np"`, `"pt"` or `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                Passed through to the attention processor.
            callback_on_step_end (`Callable`, *optional*):
                Called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*, defaults to `["latents"]`):
                Tensors from the denoising loop to hand to `callback_on_step_end`. They must be listed in the
                pipeline's `_callback_tensor_inputs`.
            output_resolution (`int`, *optional*, defaults to 1024):
                Target side length used to derive `height`/`width` and to resize condition images.
            use_kv_cache (`bool`, *optional*, defaults to `True`):
                Cache the text and condition-image keys and values after the first step. Valid because
                `causal_condition` modulates those tokens from `t = 0`, making their activations step-independent.

                Toggling this does not reproduce the same image bit-for-bit in reduced precision. Caching makes the
                decode step attend with a different sequence layout than the prefill step, so the two tile differently
                and land on different rounding; both agree with an fp32 reference to the same tolerance. A one-ULP
                difference at the first block is then amplified by 32 blocks and every sampler step, so the two
                settings give equally valid but visibly distinct samples. Fix a sample by fixing this flag.
            sample_solver (`str`, *optional*):
                WanGP's `default` schedule or one of the named Viggle Turbo profiles. Each adapter uses its matching
                trained sigma nodes.
            mask_image (`PIL.Image`, `torch.Tensor`, or `numpy.ndarray`, *optional*):
                White pixels mark the image area to regenerate for masked editing.
            inpaint_image (`PIL.Image`, `torch.Tensor`, or `numpy.ndarray`, *optional*):
                Full-resolution source image used to preserve unmasked pixels. Its reference can also be included in
                `image` so Qwen sees it while encoding the prompt.
            denoising_strength (`float`, *optional*):
                Fraction of the ordinary inpaint schedule spent denoising the source image.
            masking_strength (`float`, *optional*):
                Fraction of inpaint steps that reapply the known source outside the mask.
            model_mode (`int`, *optional*):
                0 for masked denoising, or 2/3/4/5 for LanPaint's 2/5/10/15 refinement steps.
            outpainting_dims (`tuple`, *optional*):
                WanGP's top/bottom/left/right padding percentages for spatial outpainting.

        Examples:

        Returns:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple` whose first
            element is a list with the generated images.
        """
        if inpaint_image is not None:
            if isinstance(inpaint_image, np.ndarray):
                inpaint_image = PILImage.fromarray(inpaint_image)
            elif not isinstance(inpaint_image, PILImage.Image):
                from shared.utils.utils import convert_tensor_to_image

                inpaint_image = convert_tensor_to_image(inpaint_image)
            if image is None:
                image = [inpaint_image]

        if image is not None:
            # The text encoder reads each condition image as vision context, so the pixels have to be there. Normalize
            # to PIL up front, and everything downstream — the aspect ratio below, the resize, the VAE — sees one type.
            image = image if isinstance(image, list) else [image]
            condition_images = []
            for img in image:
                if isinstance(img, PILImage.Image):
                    condition_images.append(img)
                elif isinstance(img, np.ndarray):
                    condition_images.append(PILImage.fromarray(img))
                elif isinstance(img, (list, tuple)):
                    raise ValueError(
                        "`image` is one flat set of condition images that applies to every prompt in the batch, so it "
                        "cannot be nested per prompt. Call the pipeline once per prompt when they need different "
                        "condition images."
                    )
                else:
                    raise ValueError(
                        f"`image` accepts a PIL image or a numpy array, or a list of either, but got "
                        f"{type(img).__name__}. Latents cannot stand in for a condition image here, because the text "
                        f"encoder has to see the image itself."
                    )
            image = condition_images
            calculated_width, calculated_height, _ = calculate_dimensions(
                output_resolution * output_resolution, image[-1].size[0] / image[-1].size[1]
            )
            height = height or calculated_height
            width = width or calculated_width
        height = height or output_resolution
        width = width or output_resolution

        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        source_location = None
        outpainting_source = None
        outpainting_user_mask = None
        if image is not None:
            image, mask_image, prompt, source_location, red_canvas, outpainting_user_mask = _prepare_red_outpainting(
                image, width, height, outpainting_dims, mask_image, prompt
            )
            if source_location is not None:
                outpainting_source = image[0]
                inpaint_image = outpainting_source
                image[0] = red_canvas
        elif outpainting_dims is not None and any(outpainting_dims):
            raise ValueError("Outpainting requires a control image or a main reference image.")

        if mask_image is not None and inpaint_image is None:
            if image is None:
                raise ValueError("Masked editing requires a source image.")
            inpaint_image = image[0]
        if source_location is not None or mask_image is None or model_mode is None:
            # WanGP sets model_mode=None when the inpainting selector is not
            # selected. The advertised masked-edit default is standard mode 0.
            # Outpainting has its own anchor schedule and never uses LanPaint,
            # so normalize that selector as well.
            model_mode = 0
        elif model_mode not in (0, 2, 3, 4, 5):
            raise ValueError("Qwen Image 2.1 inpainting mode must be 0, 2, 3, 4, or 5 for masked edits.")
        denoising_strength = min(1.0, max(0.0, float(denoising_strength)))
        masking_strength = min(1.0, max(0.0, float(masking_strength)))

        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 1. Preprocess condition images: one resize feeds both the text encoder and the VAE.
        input_image_sizes, input_images, vae_images = [], None, None
        if image is not None:
            input_images, vae_images = [], []
            for img in image:
                if hasattr(img, "mode") and img.mode != "RGBA":
                    img = img.convert("RGBA")
                image_width, image_height = img.size
                input_width, input_height, _ = calculate_dimensions(
                    output_resolution * output_resolution, image_width / image_height
                )
                input_image_sizes.append((input_width, input_height))
                input_images.append(self.image_processor.resize(img, width=input_width, height=input_height))
                vae_images.append(
                    self.image_processor.preprocess(img, width=input_width, height=input_height).unsqueeze(2)
                )

        # 2. Encode prompt
        # The mask is not part of the condition: `encode_prompt` returns `None` for it when nothing is padded, so
        # requiring it here would turn guidance off for a caller who passes that output straight back in.
        has_neg_prompt = negative_prompt is not None or negative_prompt_embeds is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no "
                f"negative_prompt is provided."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                "negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
            )

        prompt_embeds, prompt_embeds_mask, image_pad_mask = self.encode_prompt(
            image=input_images,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask, negative_image_pad_mask = self.encode_prompt(
                image=input_images,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

        # 3. Prepare latents
        num_channels_latents = self.transformer.config.in_channels
        latents, input_images_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        original_latents = noise_source = latent_mask = None
        if mask_image is not None:
            if inpaint_image is None:
                raise ValueError("Masked editing requires a source image.")
            source_width = outpainting_source.width if outpainting_source is not None else width
            source_height = outpainting_source.height if outpainting_source is not None else height
            source_pixels = self.image_processor.preprocess(
                inpaint_image.convert("RGBA"), width=source_width, height=source_height
            ).unsqueeze(2).to(device=device, dtype=self.vae.dtype)
            original_latents = self._encode_vae_image(source_pixels, generator)
            del source_pixels
            if source_location is not None:
                source_height, source_width, top, left = source_location
                if original_latents.shape[-2:] != (source_height // self.vae_scale_factor,
                                                     source_width // self.vae_scale_factor):
                    raise ValueError("Encoded outpainting source does not match the aligned crop.")
                canvas_latents = original_latents.new_zeros(
                    (original_latents.shape[0], self.latent_channels, 1,
                     height // self.vae_scale_factor, width // self.vae_scale_factor)
                )
                canvas_latents[..., top // self.vae_scale_factor:
                                (top + source_height) // self.vae_scale_factor,
                                left // self.vae_scale_factor:
                                (left + source_width) // self.vae_scale_factor].copy_(original_latents)
                original_latents = canvas_latents
            original_latents = original_latents[:, :, 0].flatten(2).transpose(1, 2).to(latents.dtype)
            if batch_size * num_images_per_prompt > original_latents.shape[0]:
                if (batch_size * num_images_per_prompt) % original_latents.shape[0] != 0:
                    raise ValueError("The source image batch cannot be expanded to the requested output batch.")
                original_latents = original_latents.repeat(
                    batch_size * num_images_per_prompt // original_latents.shape[0], 1, 1
                )
            latent_mask = _mask_to_latents(mask_image, width, height, latents.device, latents.dtype)
            if latent_mask.shape[1] != latents.shape[1]:
                raise ValueError("The inpainting mask does not match the prepared output dimensions.")
            latent_mask = latent_mask.expand(latents.shape[0], -1, -1)
            noise_source = latents.clone()

        outpainting_user_latent_mask = None
        if outpainting_user_mask is not None:
            outpainting_user_latent_mask = _mask_to_latents(
                outpainting_user_mask, width, height, latents.device, latents.dtype
            ).expand(latents.shape[0], -1, -1)

        img_shapes = [
            [
                *[
                    (1, vae_height // self.vae_scale_factor, vae_width // self.vae_scale_factor)
                    for vae_width, vae_height in input_image_sizes
                ],
                (1, height // self.vae_scale_factor, width // self.vae_scale_factor),
            ]
        ] * batch_size

        # 4. Prepare timesteps
        scheduler_overrides, profile_sigmas = qwen21_sampling_profile(sample_solver, num_inference_steps)
        if profile_sigmas is not None and sigmas is not None:
            raise ValueError("A Qwen acceleration profile supplies its own sigma schedule.")
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config, **scheduler_overrides)
        sigmas = profile_sigmas or (np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
                                    if sigmas is None else sigmas)
        mu = calculate_shift(
            latents.shape[1],
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device, sigmas=sigmas,
            **({} if not scheduler.config.use_dynamic_shifting else {"mu": mu})
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # The transformer's `img_mask` spans the joint sequence, so append one slot per 2x2 group of target latents.
        # The slots follow each mask's own batch size: `latents` is already expanded by `num_images_per_prompt`
        # while the masks are not, and the transformer reads the layout from row 0 because samples share it.
        def append_target_slots(mask):
            return torch.cat([mask, mask.new_ones(mask.shape[0], latents.shape[1] // 4)], dim=1)

        image_pad_mask = append_target_slots(image_pad_mask)
        if do_true_cfg:
            negative_image_pad_mask = append_target_slots(negative_image_pad_mask)

        # Text and condition-image keys and values are step-independent under `causal_condition`, so the first step
        # prefills them and later steps only recompute the target image's tokens.
        num_blocks = len(self.transformer.transformer_blocks)
        cache_enabled = use_kv_cache and self.transformer.config.causal_condition
        if cache_enabled:
            required, budget = reference_cache_plan(
                self.transformer, prompt_embeds, image_pad_mask,
                0 if input_images_latents is None else input_images_latents.shape[1],
                latents.shape[1], device,
                negative_prompt_embeds if do_true_cfg else None,
                negative_image_pad_mask if do_true_cfg else None,
            )
            cache_enabled = required <= budget
            if not cache_enabled:
                print(f"[Qwen Image 2.1 Memory] Reference KV cache needs {required / GIB:.1f} GB; "
                      f"safe cache budget is {budget / GIB:.1f} GB. Recomputing the prefix "
                      "each step; all references and output dimensions are preserved.")
        cond_cache = QwenImage21KVCache(num_blocks) if cache_enabled else None
        neg_cache = QwenImage21KVCache(num_blocks) if cache_enabled and do_true_cfg else None

        # 5. Denoising loop
        has_mask = latent_mask is not None
        first_step = min(int(num_inference_steps * (1 - denoising_strength)), num_inference_steps - 1) if has_mask else 0
        lanpaint = None
        if has_mask and model_mode in (2, 3, 4, 5) and source_location is None:
            from shared.inpainting.lanpaint import LanPaint

            lanpaint = LanPaint(NSteps={2: 2, 3: 5, 4: 10, 5: 15}[model_mode])
            denoising_strength = masking_strength = 1.0
            first_step = 0

        step_offset = 0 if source_location is not None else first_step
        scheduler.set_begin_index(step_offset)
        timesteps = timesteps[step_offset:]
        masked_steps = math.ceil((num_inference_steps - first_step) * masking_strength)
        if has_mask and source_location is None and first_step:
            sigma = scheduler.sigmas[first_step]
            latents = original_latents * (1 - sigma) + noise_source * sigma

        def denoise(model_latents, kv_mode, timestep):
            model_input = model_latents
            if input_images_latents is not None:
                model_input = torch.cat([input_images_latents, model_latents], dim=1)

            model_timestep = timestep.expand(model_latents.shape[0]).to(model_latents.dtype)
            with self.transformer.cache_context("cond"):
                prediction = self.transformer(
                    hidden_states=model_input,
                    timestep=model_timestep / 1000,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes,
                    img_mask=image_pad_mask,
                    attention_kwargs=self.attention_kwargs,
                    kv_cache=cond_cache,
                    kv_cache_mode=kv_mode,
                    return_dict=False,
                )[0][:, -model_latents.size(1):]
            negative_prediction = None
            if do_true_cfg:
                with self.transformer.cache_context("uncond"):
                    negative_prediction = self.transformer(
                        hidden_states=model_input,
                        timestep=model_timestep / 1000,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_mask=negative_prompt_embeds_mask,
                        img_shapes=img_shapes,
                        img_mask=negative_image_pad_mask,
                        attention_kwargs=self.attention_kwargs,
                        kv_cache=neg_cache,
                        kv_cache_mode=kv_mode,
                        return_dict=False,
                    )[0][:, -model_latents.size(1):]
            return prediction, negative_prediction

        combine_guidance = lambda positive, negative, scale, _: positive if negative is None else negative + scale * (positive - negative)
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    # `continue` would skip the step that prefills the cache and leave the next one decoding from an
                    # empty one, so stop the loop instead.
                    break

                self._current_timestep = t
                kv_mode = "extract" if (cache_enabled and i == 0) else ("cached" if cache_enabled else None)

                # LanPaint converts flow sigma to VE coordinates using
                # sigma / (1 - sigma); exact sigma=1 is singular. Run the
                # ordinary first denoising step, then begin its refinements.
                if lanpaint is not None and i < len(timesteps) - 1 and float(t) < 1000.0:
                    from shared.inpainting.lanpaint import _pack_latents, _unpack_latents

                    def pack(value):
                        return _pack_latents(
                            value.transpose(1, 2).reshape(
                                latents.shape[0], self.latent_channels, 1,
                                height // self.vae_scale_factor, width // self.vae_scale_factor
                            )
                        )

                    def unpack(value):
                        return _unpack_latents(value, height, width, self.vae_scale_factor).flatten(2).transpose(1, 2)

                    def packed_denoise(value, guidance):
                        positive, negative = denoise(unpack(value), kv_mode, t)
                        return pack(positive), None if negative is None else pack(negative)

                    latents = unpack(lanpaint(
                        packed_denoise, combine_guidance, true_cfg_scale, 1.0, pack(latents),
                        pack(original_latents), pack(noise_source), t / 1000,
                        pack(latent_mask.expand(latents.shape[0], -1, self.latent_channels)),
                        height=height, width=width, vae_scale_factor=self.vae_scale_factor,
                    ))

                noise_pred, neg_noise_pred = denoise(latents, kv_mode, t)
                noise_pred = combine_guidance(noise_pred, neg_noise_pred, true_cfg_scale, t)

                latents_dtype = latents.dtype
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug:
                    # https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

                absolute_step = step_offset + i
                if source_location is not None:
                    sigma = scheduler.sigmas[absolute_step + 1]
                    step_mask = _outpainting_step_mask(
                        source_location, width, height, sigma, latents.device, latents.dtype
                    ).expand(latents.shape[0], -1, -1)
                    if outpainting_user_latent_mask is not None and absolute_step < first_step + masked_steps:
                        step_mask = torch.maximum(step_mask, outpainting_user_latent_mask)
                    known = original_latents * (1 - sigma) + noise_source * sigma
                    latents = known * (1 - step_mask) + latents * step_mask
                elif has_mask and (absolute_step < first_step + masked_steps):
                    sigma = scheduler.sigmas[absolute_step + 1]
                    known = original_latents * (1 - sigma) + noise_source * sigma
                    latents = known * (1 - latent_mask) + latents * latent_mask

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, absolute_step, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        # The reference cache belongs to denoising. Release it before MMGP
        # brings in the VAE and its decode workspace.
        # `denoise` and LanPaint's adapter close over these cells. Rebind them
        # after denoising so the cache tensors are released before VAE decode
        # without deleting names that the nested functions resolve statically.
        cond_cache = None
        neg_cache = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents * latents_std + latents_mean
            if output_type == "uint8":
                latent_holder = [latents]
                del latents
                image = self.vae.decode_to_cpu_uint8(latent_holder)[:, :, 0]
            else:
                image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
                image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
