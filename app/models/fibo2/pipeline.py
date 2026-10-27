# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import hashlib
import json
import os
from pathlib import Path
from contextlib import nullcontext

import numpy as np
from PIL import Image
import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2VLImageProcessor

from mmgp import offload
from models.ideogram4.qwen3_vl_configuration import Qwen3VLConfig
from shared.utils import files_locator as fl
from shared.utils.text_encoder_cache import TextEncoderCache
from shared.utils.utils import convert_tensor_to_image
from shared.utils.loras_mutipliers import update_loras_slists
from .caption import Caption, VISION_MARKER
from .encoder import FiboEncoder, FiboProcessor, PROMPT_TEMPLATE
from .text_images import draw_text_image
from .assets import load_fonts
from .progress import generation_progress, GenerationAborted, set_phase_status
from .runtime import runtime
from .transformer import BriaFibo2Transformer2DModel
from .vae import FiboVAE

NEGATIVE_EDIT = '{"short_description":"","objects":[],"background_setting":"","lighting":{},"aesthetics":{},"photographic_characteristics":{},"style_medium":"","context":"","artistic_style":""}'
NEGATIVE_T2I = '{"short_description":"","objects":[{"description":"the main subject","texture":"smooth, featureless, plastic-like, airbrushed, no fine surface detail","appearance_details":"soft blurred edges, simplified shapes, no fine detail","skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],"background_setting":"a plain seamless studio backdrop, perfectly smooth gradient, blurred and featureless, no visible texture","lighting":{},"aesthetics":{"artifacts":["oversaturated colours","neon colour cast","overexposed highlights","lens flare","slight blur","motion blur","haze"]},"photographic_characteristics":{"depth_of_field":"shallow, heavy bokeh obscuring most of the frame","focus":"soft, slightly out of focus, lacking fine texture"},"style_medium":"","context":"","artistic_style":""}'

TEXT_ENCODER_FILES = (
    "fibo2_encoder_config.json",
    "fibo2_processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
VAE_FILE = "fibo2/flux2_vae_fp16.safetensors"
TEXT_ENCODER_WEIGHTS = (
    "Qwen3-VL-4B-Instruct_bf16.safetensors",
    "Qwen3-VL-4B-Instruct_int8_convrot.safetensors",
)


def _required_assets(text_encoder_folder):
    """Resolve tokenizer, encoder config and full VAE before creating models."""
    paths = [os.path.join(text_encoder_folder, name) for name in TEXT_ENCODER_FILES]
    paths.append(VAE_FILE)
    assets, missing = {}, []
    for path in paths:
        resolved = fl.locate_file(path, error_if_none=False)
        if resolved is None:
            missing.append(path)
        else:
            assets[path] = resolved
    if missing:
        items = ", ".join(missing)
        raise FileNotFoundError(
            "Fibo 2.0 is missing required companion assets: " + items + ". "
            "Download the Fibo 2.0 model files from Maestro before loading; "
            "the Qwen3-VL encoder, tokenizer, and FLUX.2 VAE are all required."
        )
    return assets


def _log_component_paths(component, paths):
    """Log the local checkpoint provenance once after a component is loaded."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    for path in paths:
        resolved = os.path.abspath(os.fspath(path))
        source = fl.describe_file_source(resolved)
        root = source.get("root")
        origin = f"{source.get('kind', 'unknown')} model root"
        if root:
            origin += f" {root}"
        print(f"[Fibo 2.0] Loaded {component}: {resolved} ({origin}).")


def _resolve_checkpoint_paths(source, label):
    """Resolve one or more local safetensors checkpoints without downloading."""
    if source is None:
        candidates = []
    elif isinstance(source, (str, os.PathLike)):
        candidates = [source]
    else:
        candidates = list(source)
    if not candidates:
        raise FileNotFoundError(
            f"No {label} checkpoint was selected. Download the model from Maestro first."
        )
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
            raise FileNotFoundError(
                f"Unable to locate the downloaded Fibo 2.0 {label} checkpoint: {candidate}. "
                "Download it from the pinned compatible checkpoint repository and verify "
                "that you can access it under the applicable model terms; Maestro does "
                "not substitute gated Bria source weights."
            )
        if not str(located).lower().endswith(".safetensors"):
            raise ValueError(f"Fibo 2.0 {label} must use safetensors: {located}")
        paths.append(os.path.abspath(located))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise FileNotFoundError(f"No safetensors files were found for Fibo 2.0 {label}.")
    return paths


def _resolve_text_encoder_checkpoint(source, folder):
    if source is not None:
        return _resolve_checkpoint_paths(source, "Qwen3-VL encoder")
    for name in TEXT_ENCODER_WEIGHTS:
        located = fl.locate_file(os.path.join(folder, name), error_if_none=False)
        if located:
            return [located]
    raise FileNotFoundError(
        "Fibo 2.0 is missing its required Qwen3-VL-4B encoder weights. "
        "Download the model from Maestro first; if access to the selected "
        "checkpoint is unavailable, verify the repository and its terms."
    )


def _autocast(device, dtype):
    device = torch.device(device)
    return torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()


def _pil_image(image):
    if isinstance(image, Image.Image):
        return image.copy()
    if torch.is_tensor(image):
        return convert_tensor_to_image(image)
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return source.copy()
    return Image.fromarray(np.asarray(image))


def _positive_integer(label, value):
    if isinstance(value, bool):
        raise ValueError(f"Fibo 2.0 {label} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Fibo 2.0 {label} must be a positive integer.") from error
    if integer <= 0 or str(value).strip() not in (str(integer), f"{integer}.0"):
        raise ValueError(f"Fibo 2.0 {label} must be a positive integer.")
    return integer


def validate_generation_inputs(width, height, sampling_steps, batch_size, references=(), has_mask=False,
                               edit_only=False):
    width = _positive_integer("width", width)
    height = _positive_integer("height", height)
    sampling_steps = _positive_integer("sampling steps", sampling_steps)
    batch_size = _positive_integer("batch size", batch_size)
    if width > 8192 or height > 8192:
        raise ValueError("Fibo 2.0 supports dimensions up to 8192 pixels per side.")
    if width % 16 or height % 16:
        raise ValueError("Fibo 2.0 width and height must be multiples of 16 pixels.")
    if len(references) > 4:
        raise ValueError("Fibo 2.0 supports up to four ordered reference images.")
    if has_mask and len(references) != 1:
        raise ValueError("Fibo 2.0 masked editing requires exactly one source image.")
    if edit_only and not references:
        raise ValueError("Fibo 2.0 Edit Turbo requires a source image. Use Dev for text-only generation.")
    return width, height, sampling_steps, batch_size


def pack(x):
    b, c, h, w = x.shape
    return x.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpack(x):
    b, c, h, w = x.shape
    return x.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


class FiboPipeline:
    def __init__(self, transformer, encoder, processor, vae, edit_only=False):
        self.transformer, self.text_encoder, self.processor, self.vae = transformer, encoder, processor, vae
        self.edit_only = edit_only
        self.text_encoder_cache = TextEncoderCache(max_size_mb=768)
        self.preview_mean = vae.bn.running_mean.detach().float().cpu().reshape(1, 128, 1, 1)
        self.preview_scale = (vae.bn.running_var.detach().float().cpu() + vae.bn_eps).rsqrt().reshape(1, 128, 1, 1)
        self._interrupt = False

    @property
    def _interrupt(self):
        return self.transformer._interrupt

    @_interrupt.setter
    def _interrupt(self, value):
        self.transformer._interrupt = self.transformer.perceiver._interrupt = value
        self.text_encoder._interrupt = self.text_encoder.language_model._interrupt = self.text_encoder.visual._interrupt = value
        self.vae._interrupt = bool(value)

    def abort(self):
        self._interrupt = True

    def encode(self, prompt, images, device):
        keys = tuple((image.size, hashlib.sha256(image.tobytes()).digest()) for image in images)
        def compute(items):
            values = []
            for text in items:
                with torch.device("cpu"):
                    inputs = self.processor(text=[PROMPT_TEMPLATE.format(text)], images=images or None, padding="longest",
                                            max_length=4096, truncation=True, return_tensors="pt")
                with _autocast(device, torch.bfloat16):
                    values.append(self.text_encoder(inputs))
                if self._interrupt:
                    raise GenerationAborted
            return values
        return self.text_encoder_cache.encode(compute, [prompt], device=device, cache_keys=(prompt, keys))[0]

    def preview(self, latents):
        value = pack(latents)
        value = value / self.preview_scale.to(value) + self.preview_mean.to(value)
        return unpack(value).transpose(0, 1)

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt="", n_prompt="", seed=0, sampling_steps=50, width=1024, height=1024, guide_scale=5.0,
                 batch_size=1, input_ref_images=None, input_frames=None, input_masks=None, video_prompt_type="",
                 callback=None, VAE_tile_size=None, set_progress_status=None, loras_slists=None, **kwargs):
        device = runtime.execution_device(self.transformer)
        text = "\n".join(line for line in input_prompt.splitlines() if not line.lstrip().startswith("#"))
        if "I" in video_prompt_type:
            if input_ref_images is None:
                references = []
            elif isinstance(input_ref_images, (list, tuple)):
                references = list(input_ref_images)
            else:
                references = [input_ref_images]
        else:
            references = []
        if input_frames is not None:
            references.insert(0, _pil_image(input_frames))
        references = [_pil_image(image).convert("RGB") for image in references if image is not None]
        if input_masks is not None:
            input_masks = (
                convert_tensor_to_image(input_masks, mask_levels=True)
                if torch.is_tensor(input_masks) else _pil_image(input_masks)
            ).convert("L")
        width, height, sampling_steps, batch_size = validate_generation_inputs(
            width, height, sampling_steps, batch_size, references,
            has_mask=input_masks is not None, edit_only=self.edit_only,
        )
        if self.edit_only:
            guide_scale = 1.0
        if input_masks is not None:
            if input_masks.size != references[0].size:
                input_masks = input_masks.resize(references[0].size, Image.Resampling.NEAREST)
            binary = input_masks.point(lambda value: 255 if value >= 128 else 0)
            references[0] = Image.composite(Image.new("RGB", references[0].size, (128, 128, 128)), references[0], binary)
        caption, _ = Caption._parse_prompt(text)
        text_images = []
        if isinstance(caption, dict):
            entries = [entry["text"] for entry in caption.get("text_render", [])
                       if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]]
            if entries:
                fonts = load_fonts()
                text_images = [draw_text_image(entry, fonts) for entry in entries]
        vision_images = []
        for image in references:
            thumbnail = image.copy()
            thumbnail.thumbnail((768, 768))
            vision_images.append(thumbnail)
        base_prompt = None
        if references:
            prompt = Caption._get_edit_prompt(text, len(references), bool(text_images))
        else:
            if isinstance(caption, dict):
                if text_images:
                    base_prompt = json.dumps(Caption._mark_text_images(caption.copy(), ""), separators=(",", ":"), ensure_ascii=False)
                    caption = Caption._mark_text_images(caption, VISION_MARKER)
                prompt = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
            else:
                prompt = caption or ""
        set_phase_status("Preparing Text Conditioning")
        positive = self.encode(prompt, vision_images + text_images, device)
        base = self.encode(base_prompt, [], device) if base_prompt is not None else positive
        negative = self.encode(n_prompt or (NEGATIVE_EDIT if references else NEGATIVE_T2I), [], device) if guide_scale > 1 else None
        set_phase_status("Preparing Image Conditioning")
        tiled, tile_size = VAE_tile_size if VAE_tile_size is not None else (True, 1024)
        contexts = []
        for image in references:
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float()
            pixels = (pixels / 127.5 - 1).to(runtime.execution_device(self.vae))
            with _autocast(runtime.execution_device(self.vae), torch.float16):
                encoded = self.vae.encode(pixels, tile_size=tile_size if tiled else 0)
                contexts.append(unpack(encoded).to(device, dtype=torch.bfloat16))
            del pixels
            if self._interrupt:
                raise GenerationAborted
        if batch_size > 1:
            positive, base = positive.repeat(batch_size, 1, 1, 1), base.repeat(batch_size, 1, 1, 1)
            negative = None if negative is None else negative.repeat(batch_size, 1, 1, 1)
            contexts = [value.repeat(batch_size, 1, 1, 1) for value in contexts]
        set_phase_status("Preparing Timesteps")
        with torch.device("cpu"):
            scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, shift=3.0)
        tokens = (height // 16) * (width // 16)
        mu = (1.15 - 0.5) / (4096 - 256) * (tokens - 256) + 0.5
        scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / sampling_steps, sampling_steps), device=device, mu=mu)
        latents = torch.cat([torch.randn((1, 32, height // 8, width // 8), generator=torch.Generator(device="cpu").manual_seed(seed + i),
                                       device="cpu", dtype=torch.float32) for i in range(batch_size)]).to(device)
        # Read branching scalars once on CPU, outside graph capture and the hot loop.
        times = scheduler.timesteps.cpu().tolist()
        update_loras_slists(self.transformer, loras_slists, sampling_steps)
        if callback is not None:
            callback(-1, None, True)
        with _autocast(device, torch.bfloat16):
            for step, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising", mininterval=1/3)):
                if self._interrupt:
                    raise GenerationAborted
                offload.set_step_no_for_lora(self.transformer, step)
                model_input = latents.to(torch.bfloat16)
                timestep = t.expand(batch_size) / 1000
                embedding = positive if times[step] / 1000 <= 0.9 else base
                velocity = self.transformer(model_input, timestep, embedding, context_latents=contexts)[0].float()
                if self._interrupt:
                    raise GenerationAborted
                if negative is not None:
                    unconditional = self.transformer(model_input, timestep, negative, context_latents=contexts)[0].float()
                    velocity = unconditional + guide_scale * (velocity - unconditional)
                    del unconditional
                latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
                del velocity, model_input
                if callback is not None:
                    callback(step, self.preview(latents), False)
        del positive, negative, base, contexts, embedding
        set_phase_status("VAE Decoding")
        with _autocast(runtime.execution_device(self.vae), torch.float16):
            images = [self.vae(pack(item.unsqueeze(0)).to(runtime.execution_device(self.vae)), tile_size=tile_size if tiled else 0)
                      for item in latents]
        return torch.cat(images).transpose(0, 1)


def load_pipeline(model_filename, text_encoder_filename, model_def, model_type, save_quantized):
    architecture = model_def.get("architecture")
    if architecture not in ("fibo2_dev", "fibo2_edit_turbo"):
        raise ValueError(f"Unsupported Fibo 2.0 architecture: {architecture!r}.")
    folder = model_def.get("text_encoder_folder", "Qwen3-VL-4B-Instruct")
    assets = _required_assets(folder)
    transformer_paths = _resolve_checkpoint_paths(model_filename, "transformer")
    text_encoder_paths = _resolve_text_encoder_checkpoint(text_encoder_filename, folder)
    config_path = Path(__file__).with_name(f"{architecture}.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    encoder_config = Qwen3VLConfig.from_json_file(assets[os.path.join(folder, "fibo2_encoder_config.json")])
    with init_empty_weights(include_buffers=True):
        transformer = BriaFibo2Transformer2DModel(**config)
        encoder = FiboEncoder(encoder_config)
        vae = FiboVAE()
    encoder.visual.rotary_pos_emb.reset_inv_freq()
    encoder.language_model.rotary_emb.reset_inv_freq()
    offload.load_model_data(transformer, transformer_paths, writable_tensors=False, default_dtype=torch.bfloat16)
    _log_component_paths("transformer checkpoint", transformer_paths)
    offload.load_model_data(encoder, text_encoder_paths, writable_tensors=False, default_dtype=torch.bfloat16,
                           fused_split_map=encoder.split_linear_modules_map)
    _log_component_paths("Qwen3-VL encoder", text_encoder_paths)
    offload.load_model_data(vae, assets[VAE_FILE], writable_tensors=False, default_dtype=None)
    _log_component_paths("FLUX.2 VAE", assets[VAE_FILE])
    for module in (transformer, encoder, vae):
        module.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(assets[os.path.join(folder, "tokenizer.json")]).parent),
        local_files_only=True, extra_special_tokens={},
    )
    image_config = json.loads(Path(assets[os.path.join(folder, "fibo2_processor_config.json")]).read_text(encoding="utf-8"))["image_processor"]
    image_processor = Qwen2VLImageProcessor(**image_config)
    processor = FiboProcessor(image_processor, tokenizer)
    if save_quantized:
        from wgp import save_quantized_model
        save_quantized_model(transformer, model_type, transformer_paths[0], torch.bfloat16, str(config_path),
                             convrot_prefixes=("layers.", "noise_refiner.", "perceiver.layers."))
    return FiboPipeline(transformer, encoder, processor, vae, edit_only=architecture == "fibo2_edit_turbo")
