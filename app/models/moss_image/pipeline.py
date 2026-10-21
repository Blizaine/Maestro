# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Native text-to-image pipeline for MOSS-Image 5B."""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re
from numbers import Real

import torch
from accelerate import init_empty_weights
from mmgp import offload
from tqdm import tqdm
from transformers import AutoTokenizer

from shared.utils import files_locator as fl
from shared.utils.text_encoder_cache import TextEncoderCache

from .progress import GenerationAborted, generation_progress, set_phase_status
from .runtime import runtime
from .text_encoder import TextEncoder, Qwen3_5TextConfig, encode_prompt
from .transformer import MOSSImageTransformer, rope_embed
from .vae import MOSSImageVAE


MAX_PIXELS = 4_194_304
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 3.0
DEFAULT_FLOW_SHIFT = 5.0
TEXT_ENCODER_FOLDER = "Qwen3.5-9B"
VAE_CHECKPOINT = os.path.join("moss_image", "flux2_small_decoder_bf16.safetensors")


def sampling_schedule(steps, shift, device="cpu"):
    """Construct MOSS's nearest-training-grid shifted flow schedule in CPU FP32."""
    steps = _positive_integer("Sampling steps", steps)
    try:
        shift = float(shift)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Flow shift must be a positive finite number.") from error
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("Flow shift must be a positive finite number.")

    lower = shift / (1000 + shift - 1)
    values = torch.linspace(1, lower, steps, device="cpu", dtype=torch.float32)
    sigmas = shift * values / (1 + (shift - 1) * values)
    training = torch.linspace(1, 0.001, 1000, device="cpu", dtype=torch.float32).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=999)
    lower_index = (upper - 1).clamp(min=0)
    nearest = torch.where(
        (sigmas - training[lower_index]).abs() <= (training[upper] - sigmas).abs(),
        lower_index,
        upper,
    )
    schedule = torch.cat((torch.unique_consecutive(training[nearest]), torch.zeros(1, device="cpu")))
    return schedule.to(device)


def _positive_integer(label, value):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer.")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.isdecimal():
            raise ValueError(f"{label} must be a positive integer.")
        value = int(stripped)
    elif isinstance(value, Real):
        if not math.isfinite(float(value)) or int(value) != value:
            raise ValueError(f"{label} must be a positive integer.")
        value = int(value)
    else:
        raise ValueError(f"{label} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _has_value(value):
    if value is None:
        return False
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return bool(value)
    return True


def _read_dimensions(inputs):
    width, height = inputs.get("width"), inputs.get("height")
    if width is None and height is None:
        resolution = inputs.get("resolution")
        match = re.fullmatch(r"\s*(\d+)\s*[x×]\s*(\d+)\s*", str(resolution or ""), re.IGNORECASE)
        if match:
            width, height = match.groups()
    if width is None and height is None:
        return None
    if width is None or height is None:
        raise ValueError("MOSS Image needs both width and height.")
    return _positive_integer("Image width", width), _positive_integer("Image height", height)


def validate_generation_inputs(inputs):
    """Return a useful validation message, or None for supported text-to-image settings."""
    inputs = dict(inputs or {})
    mode = inputs.get("image_mode", inputs.get("model_mode"))
    if mode is not None and str(mode).strip() not in ("1", "1.0"):
        return "MOSS Image supports text-to-image only; select the standard image generation mode."

    if str(inputs.get("video_prompt_type") or "").strip():
        return "MOSS Image is text-to-image only and does not accept reference images. Clear the reference-image selection."

    unsupported_inputs = (
        "image_refs", "input_ref_images", "original_input_ref_images", "image_mask",
        "input_masks", "input_frames", "input_video", "image_start", "audio_prompt",
        "audio_refs", "input_audio", "outpainting_dims",
    )
    if any(_has_value(inputs.get(key)) for key in unsupported_inputs):
        return "MOSS Image is text-to-image only and does not accept image, mask, video, or audio inputs."

    try:
        dimensions = _read_dimensions(inputs)
    except ValueError as error:
        return str(error)
    if dimensions is not None:
        width, height = dimensions
        if width % 16 or height % 16:
            return "MOSS Image width and height must each be multiples of 16."
        if width * height > MAX_PIXELS:
            return "MOSS Image supports canvases up to 4 megapixels (4,194,304 pixels)."

    for key, label in (("batch_size", "Batch size"), ("sampling_steps", "Sampling steps"),
                       ("num_inference_steps", "Sampling steps")):
        if key in inputs and inputs[key] is not None:
            try:
                _positive_integer(label, inputs[key])
            except ValueError as error:
                return str(error)

    shift = inputs.get("flow_shift", inputs.get("shift"))
    if shift is not None:
        try:
            shift = float(shift)
        except (TypeError, ValueError, OverflowError):
            return "Flow shift must be a positive finite number."
        if not math.isfinite(shift) or shift <= 0:
            return "Flow shift must be a positive finite number."

    guidance = inputs.get("guidance_scale", inputs.get("guide_scale"))
    if guidance is not None:
        try:
            guidance = float(guidance)
        except (TypeError, ValueError, OverflowError):
            return "Guidance scale must be a finite number."
        if not math.isfinite(guidance):
            return "Guidance scale must be a finite number."

    solver = str(inputs.get("sample_solver") or "default").strip().lower()
    if solver not in ("", "default", "euler"):
        return "MOSS Image uses its native flow-matching Euler sampler."
    return None


def _prompt_without_comment_lines(prompt):
    return "\n".join(line for line in str(prompt or "").splitlines() if not line.lstrip().startswith("#"))


def _autocast(device):
    if torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _unpack_vae_tile_size(setting):
    """Normalize WGP's ``(enabled, output_pixel_size)`` VAE tile choice."""
    if setting is None:
        return True, 1024
    if isinstance(setting, (tuple, list)):
        if not setting:
            return False, 0
        if len(setting) == 1:
            return bool(setting[0]), 1024 if setting[0] else 0
        return bool(setting[0]), _positive_integer("VAE tile size", setting[1]) if setting[0] else 0
    if isinstance(setting, bool):
        return setting, 1024 if setting else 0
    tile_size = _positive_integer("VAE tile size", setting)
    return True, tile_size


class MOSSImagePipeline:
    def __init__(self, transformer, text_encoder, tokenizer, vae):
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.text_encoder_cache = TextEncoderCache()
        self.preview_mean = vae.bn.running_mean.detach().float().cpu().reshape(1, 128, 1, 1)
        self.preview_std = (vae.bn.running_var.detach().float().cpu() + vae.bn_eps).sqrt().reshape(1, 128, 1, 1)

    @property
    def _interrupt(self):
        return bool(
            getattr(self.transformer, "_interrupt", False)
            or getattr(self.text_encoder, "_interrupt", False)
            or getattr(self.vae, "_interrupt", False)
        )

    @_interrupt.setter
    def _interrupt(self, value):
        value = bool(value)
        self.transformer._interrupt = value
        self.text_encoder._interrupt = value
        self.vae._interrupt = value

    def abort(self):
        self._interrupt = True

    def preview_latents(self, latents):
        z = latents * self.preview_std.to(latents) + self.preview_mean.to(latents)
        batch, channels, height, width = z.shape
        return z.reshape(batch, channels // 4, 2, 2, height, width).permute(0, 1, 4, 2, 5, 3).reshape(
            batch, channels // 4, 2 * height, 2 * width
        ).transpose(0, 1)

    @generation_progress
    @torch.inference_mode()
    def generate(
        self,
        input_prompt="",
        n_prompt="",
        seed=0,
        sampling_steps=DEFAULT_STEPS,
        width=1024,
        height=1024,
        guide_scale=DEFAULT_GUIDANCE,
        flow_shift=DEFAULT_FLOW_SHIFT,
        shift=None,
        batch_size=1,
        callback=None,
        VAE_tile_size=None,
        set_progress_status=None,
        video_prompt_type="",
        image_mode=1,
        model_mode=None,
        sample_solver="default",
        **kwargs,
    ):
        if shift is not None:
            flow_shift = shift
        settings = {
            **kwargs,
            "width": width,
            "height": height,
            "batch_size": batch_size,
            "sampling_steps": sampling_steps,
            "flow_shift": flow_shift,
            "guidance_scale": guide_scale,
            "video_prompt_type": video_prompt_type,
            "image_mode": image_mode if model_mode is None else model_mode,
            "sample_solver": sample_solver,
        }
        validation_error = validate_generation_inputs(settings)
        if validation_error:
            raise ValueError(validation_error)

        width = _positive_integer("Image width", width)
        height = _positive_integer("Image height", height)
        batch_size = _positive_integer("Batch size", batch_size)
        sampling_steps = _positive_integer("Sampling steps", sampling_steps)
        guide_scale = float(guide_scale)
        flow_shift = float(flow_shift)
        device = runtime.execution_device(self.transformer)
        prompts = [str(n_prompt or ""), str(input_prompt or "")] if guide_scale != 1 else [str(input_prompt or "")]
        prompts = [_prompt_without_comment_lines(prompt) for prompt in prompts]

        set_phase_status("Preparing Text Conditioning")
        texts = self.text_encoder_cache.encode(
            lambda items: [encode_prompt(self.text_encoder, self.tokenizer, prompt) for prompt in items],
            prompts,
            device=device,
        )
        set_phase_status("Preparing Timesteps")
        schedule = sampling_schedule(sampling_steps, flow_shift, device)
        loras_slists = kwargs.get("loras_slists")
        if loras_slists is not None:
            from shared.utils.loras_mutipliers import update_loras_slists

            update_loras_slists(self.transformer, loras_slists, len(schedule) - 1)
        latent_height, latent_width = height // 16, width // 16
        config = self.transformer.config
        ropes = [
            rope_embed(latent_height, latent_width, text.shape[1], device,
                       config.dim // config.num_heads, config.rope_base).to(dtype=torch.bfloat16)
            for text in texts
        ]

        latents = torch.empty((batch_size, 128, latent_height, latent_width), device="cpu", dtype=torch.float32)
        for image_no in range(batch_size):
            generator = torch.Generator(device="cpu").manual_seed(int(seed or 0) + image_no)
            torch.randn((1, 128, latent_height, latent_width), generator=generator,
                        device="cpu", dtype=torch.float32, out=latents[image_no:image_no + 1])
        latents = latents.to(device)
        if callback is not None:
            callback(-1, None, True)

        with _autocast(device):
            for step in tqdm(range(len(schedule) - 1), desc="Denoising", mininterval=1 / 3):
                if self._interrupt:
                    raise GenerationAborted
                if loras_slists is not None:
                    offload.set_step_no_for_lora(self.transformer, step)
                predictions = self.transformer(latents, schedule[step] * 1000, texts, ropes)
                if self._interrupt:
                    raise GenerationAborted
                velocity = (
                    predictions[0] + guide_scale * (predictions[1] - predictions[0])
                    if len(texts) == 2 else predictions[0]
                )
                latents = latents + (schedule[step + 1] - schedule[step]) * velocity
                del predictions, velocity
                if callback is not None:
                    callback(step, self.preview_latents(latents), False)

        if self._interrupt:
            raise GenerationAborted
        set_phase_status("VAE Decoding")
        tiled, tile_size = _unpack_vae_tile_size(VAE_tile_size)
        with _autocast(device):
            pixels = self.vae(
                latents.to(device=runtime.execution_device(self.vae), dtype=torch.bfloat16),
                tile_size=tile_size if tiled else 0,
            )
        if self._interrupt:
            raise GenerationAborted
        return pixels.transpose(0, 1)


def _resolve_checkpoint_paths(source, label):
    if isinstance(source, (list, tuple)):
        candidates = list(source)
    elif source is None:
        candidates = []
    else:
        candidates = [source]
    if not candidates:
        raise ValueError(f"No {label} checkpoint was provided.")

    paths = []
    for candidate in candidates:
        candidate = os.fspath(candidate)
        if os.path.isdir(candidate):
            paths.extend(str(path) for path in sorted(Path(candidate).glob("*.safetensors")))
            continue
        located = candidate if os.path.isfile(candidate) else fl.locate_file(candidate, error_if_none=False)
        if located is None:
            located = fl.locate_file(os.path.basename(candidate), error_if_none=False)
        if located is None:
            raise FileNotFoundError(f"Unable to locate the downloaded {label} checkpoint: {candidate}")
        if not str(located).lower().endswith(".safetensors"):
            raise ValueError(f"{label} must use a safetensors checkpoint: {located}")
        paths.append(os.path.abspath(located))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise FileNotFoundError(f"No safetensors files were found for {label}.")
    return paths


def _required_assets(folder):
    from models.moss_image.moss_image_handler import TOKENIZER_FILES, VAE_FILE

    paths = [
        os.path.join(folder, name) for name in TOKENIZER_FILES
    ] + [os.path.join("moss_image", VAE_FILE)]
    resolved, missing = {}, []
    for path in paths:
        located = fl.locate_file(path, error_if_none=False)
        if located is None:
            missing.append(path)
        else:
            resolved[path] = located
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"MOSS Image is missing required assets: {joined}. Download the model files before loading it.")
    return resolved


def load_pipeline(model_filename, text_encoder_filename, model_def, model_type, save_quantized, dtype):
    config_path = Path(__file__).with_name("moss_image.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_folder = model_def.get("text_encoder_folder", TEXT_ENCODER_FOLDER)
    assets = _required_assets(text_folder)
    text_config = Qwen3_5TextConfig.from_json_file(assets[os.path.join(text_folder, "config.json")])
    tokenizer_path = str(Path(assets[os.path.join(text_folder, "tokenizer_config.json")]).parent)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        padding_side="right",
        extra_special_tokens={},
    )

    transformer_paths = _resolve_checkpoint_paths(model_filename, "MOSS Image transformer")
    if text_encoder_filename is None:
        from .moss_image_handler import TEXT_ENCODER_FILES

        candidates = [fl.locate_file(os.path.join(text_folder, name), error_if_none=False)
                      for name in reversed(TEXT_ENCODER_FILES)]
        text_encoder_filename = next((path for path in candidates if path), None)
    text_encoder_paths = _resolve_checkpoint_paths(text_encoder_filename, "MOSS Image text encoder")
    vae_path = assets[os.path.join("moss_image", "flux2_small_decoder_bf16.safetensors")]

    with init_empty_weights(include_buffers=True):
        transformer = MOSSImageTransformer(**config)
        encoder = TextEncoder(text_config)
        vae = MOSSImageVAE()

    with torch.device("cpu"):
        encoder.rotary_emb.inv_freq, encoder.rotary_emb.attention_scaling = (
            encoder.rotary_emb.compute_default_rope_parameters(text_config, device="cpu")
        )
    encoder.rotary_emb.original_inv_freq = encoder.rotary_emb.inv_freq.clone()
    encoder.rotary_emb._lock_dtype = torch.float32

    offload.load_model_data(
        transformer,
        transformer_paths,
        writable_tensors=False,
        default_dtype=torch.bfloat16,
        fused_split_map=transformer.split_linear_modules_map,
    )
    offload.load_model_data(
        encoder,
        text_encoder_paths,
        writable_tensors=False,
        default_dtype=torch.bfloat16,
        fused_split_map=encoder.split_linear_modules_map,
    )
    offload.load_model_data(vae, [vae_path], writable_tensors=False, default_dtype=None)
    for module in (transformer, encoder, vae):
        module.eval()

    if save_quantized:
        from wgp import save_quantized_model

        save_quantized_model(transformer, model_type, transformer_paths[0], dtype, str(config_path))
    return MOSSImagePipeline(transformer, encoder, tokenizer, vae)
