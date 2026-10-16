"""MMGP-backed inference runtime for distilled FLUX 3 Image Klein."""

import os
import re
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from accelerate import init_empty_weights
from PIL import Image
from safetensors import safe_open

from mmgp import offload
from shared.utils import files_locator as fl

from ..qwen21.memory import configure_encoder_attention
from .autoencoder import AutoEncoder, latent_ids, latent_shape
from .dit import DiT
from .pipeline import (
    MAX_REFERENCES,
    pack,
    prepare_multi_reference,
    prepare_reference,
    scaled_ids,
    to_tensor,
)
from .sampling import DEFAULT_ALPHA, DEFAULT_STEPS, denoise_flow_map, timesteps
from .text_encoder import FEATURE_WIDTH, OUTPUT_LAYERS, encode_prompt


COMPONENT_ROOT = "flux3_klein"
TEXT_ENCODER_FOLDER = os.path.join(COMPONENT_ROOT, "text_encoder")
TEXT_ENCODER_SHARDS = (
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)
VAE_CHECKPOINT = os.path.join(COMPONENT_ROOT, "ae.safetensors")


class GenerationCancelled(Exception):
    """Internal signal used to stop a generation at a safe stage boundary."""


def _resolve_local_checkpoint(source, label: str) -> list[str]:
    """Resolve one or more already-downloaded checkpoint files without fetching data."""
    if isinstance(source, (tuple, list)):
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
        located = candidate if os.path.isfile(candidate) else None
        if located is None:
            located = fl.locate_file(candidate, error_if_none=False)
        if located is None:
            located = fl.locate_file(os.path.basename(candidate), error_if_none=False)
        if located is None:
            raise FileNotFoundError(f"Unable to locate downloaded {label} checkpoint: {candidate}")
        if not str(located).lower().endswith(".safetensors"):
            raise ValueError(f"{label} must use a native safetensors checkpoint: {located}")
        paths.append(os.path.abspath(located))

    unique = list(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError(f"No safetensors files were found for {label}.")
    return unique


def _expand_numbered_shards(paths: list[str], label: str) -> list[str]:
    """Expand a first-shard path to its complete local safetensors set."""
    expanded = []
    for path in paths:
        match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.safetensors", os.path.basename(path))
        if not match:
            expanded.append(path)
            continue
        stem, _, total_text = match.groups()
        total = int(total_text)
        parent = os.path.dirname(path)
        shards = [os.path.join(parent, f"{stem}-{index:05d}-of-{total:05d}.safetensors")
                  for index in range(1, total + 1)]
        missing = [shard for shard in shards if not os.path.isfile(shard)]
        if missing:
            raise FileNotFoundError(f"{label} checkpoint is missing shard(s): {missing[:3]}")
        expanded.extend(shards)
    return list(dict.fromkeys(expanded))


def _mmgp_source(paths: list[str]):
    """Pass one member of MMGP-auto-expanded shards to avoid rereading a set."""
    if len(paths) <= 1:
        return paths[0]
    shard_matches = [re.fullmatch(
        r"(.+)-(\d{5})-of-(\d{5})\.safetensors", os.path.basename(path)
    ) for path in paths]
    if all(shard_matches):
        families = {(os.path.dirname(path), match.group(1), match.group(3))
                    for path, match in zip(paths, shard_matches)}
        if len(families) == 1:
            # MMGP expands a -of-N path itself, so feeding it every shard would
            # reread the complete encoder N times.
            return paths[0]
    return paths


def _load_mmgp_checkpoint(module: torch.nn.Module, paths: list[str], **kwargs) -> None:
    """Load one local component while avoiding redundant numbered-shard reads."""
    offload.load_model_data(module, _mmgp_source(paths), **kwargs)


def _validate_checkpoint_shapes(module: torch.nn.Module, paths: list[str], label: str,
                                remap: Callable[[str, tuple[int, ...]], str | None] | None = None) -> None:
    """Require exact tensor names and shapes before MMGP materializes weights."""
    expected = {name: tuple(tensor.shape) for name, tensor in module.state_dict().items()}
    actual: dict[str, tuple[int, ...]] = {}
    for path in paths:
        try:
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                for raw_name in checkpoint.keys():
                    shape = tuple(checkpoint.get_slice(raw_name).get_shape())
                    name = remap(raw_name, shape) if remap is not None else raw_name
                    if name is None:
                        continue
                    if name in actual:
                        raise ValueError(f"{label} checkpoint repeats tensor key {name!r}.")
                    actual[name] = shape
        except Exception as error:
            if isinstance(error, ValueError) and str(error).startswith(label):
                raise
            raise ValueError(f"Unable to inspect {label} checkpoint {path}: {error}") from error

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        (name, expected[name], actual[name])
        for name in set(expected) & set(actual)
        if expected[name] != actual[name]
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing keys {missing[:5]}")
        if unexpected:
            details.append(f"unexpected keys {unexpected[:5]}")
        if mismatched:
            details.append(f"shape mismatches {mismatched[:5]}")
        raise ValueError(f"{label} checkpoint does not match the native FLUX 3 architecture: " + "; ".join(details))


def _qwen_checkpoint_key(raw_name: str, shape: tuple[int, ...], config) -> str | None:
    """Map official causal-LM keys to the bare Qwen3VLModel; validate its omitted LM head."""
    if raw_name.startswith("model."):
        return raw_name[len("model."):]
    if raw_name == "lm_head.weight":
        expected = (int(config.text_config.vocab_size), int(config.text_config.hidden_size))
        if shape != expected:
            raise ValueError(f"FLUX 3 Qwen3-VL lm_head has shape {shape}; expected {expected}.")
        return None
    raise ValueError(f"Unexpected Qwen3-VL checkpoint key outside model./lm_head.weight: {raw_name}")


def _strip_qwen_checkpoint_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop only the unused vocabulary head and remove the official `model.` prefix."""
    mapped = {}
    for raw_name, tensor in state_dict.items():
        if raw_name.startswith("model."):
            mapped[raw_name[len("model."):]] = tensor
        elif raw_name == "lm_head.weight":
            continue
        else:
            raise ValueError(f"Unexpected Qwen3-VL checkpoint key outside model./lm_head.weight: {raw_name}")
    return mapped


def _has_active_loras(value) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, dict):
        phases = ("phase1", "phase2", "phase3")
        if any(phase in value for phase in phases):
            return any(_has_active_loras(value.get(phase)) for phase in phases)
        return any(_has_active_loras(item) for item in value.values())
    if isinstance(value, (tuple, list, set)):
        return any(_has_active_loras(item) for item in value)
    return bool(value)


def _as_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.copy()
    if torch.is_tensor(image):
        from shared.utils.utils import convert_tensor_to_image
        return convert_tensor_to_image(image)
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as source:
            return source.copy()
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    raise TypeError(f"Unsupported reference-image type: {type(image).__name__}")


def _active_references(video_prompt_type: str, input_ref_images,
                       original_input_ref_images) -> list[Image.Image]:
    selector = str(video_prompt_type or "").strip()
    if any(letter != "I" for letter in selector):
        raise ValueError("FLUX 3 Image Klein supports reference-image editing only; select Reference images or None.")
    references = []
    if "I" in selector:
        selected = original_input_ref_images
        if selected is None or (isinstance(selected, (list, tuple)) and not selected):
            selected = input_ref_images
        if selected is not None:
            selected = selected if isinstance(selected, (list, tuple)) else [selected]
            references.extend(_as_pil(image) for image in selected)
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"FLUX 3 Image Klein supports up to {MAX_REFERENCES} reference images.")
    return references


def _active_outpainting(dims) -> bool:
    """Detect active margins while accepting WGP's serialized all-zero defaults."""
    if dims is None:
        return False
    if isinstance(dims, str):
        values = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", dims)
    elif torch.is_tensor(dims):
        values = dims.detach().reshape(-1).tolist()
    elif isinstance(dims, (list, tuple, np.ndarray)):
        values = np.asarray(dims).reshape(-1).tolist()
    else:
        return False
    try:
        return any(float(value) > 0 for value in values)
    except (TypeError, ValueError):
        return False


def _lock_native_vae_fp32(vae: AutoEncoder) -> None:
    """Keep VAE parameters and running-stat buffers in their native FP32 dtype."""
    for module in vae.modules():
        has_parameters = any(True for _ in module.parameters(recurse=False))
        has_buffers = any(True for _ in module.buffers(recurse=False))
        if has_parameters or has_buffers:
            module._lock_dtype = torch.float32
    # MMGP consults the root lock for buffers (including VAE running stats).
    vae._lock_dtype = torch.float32
    vae._convertWeightsFloatTo = torch.float32


def _configure_native_vae_offload(vae: AutoEncoder) -> None:
    """Tell MMGP to retain FP32 VAE weights and wrap both large forwards."""
    vae._model_dtype = torch.float32
    vae._offload_hooks = ["encode", "decode"]


def _report_batch_start(callback, batch_size: int, steps: int) -> None:
    if callable(callback) and batch_size > 1:
        callback(-1, None, True, override_num_inference_steps=steps * batch_size)


def _make_step_callback(callback, batch_index: int, steps: int, check_cancel):
    def step_end(step):
        check_cancel()
        if callable(callback):
            callback(batch_index * steps + step, None, False)

    return step_end


def _check_model_size(width, height) -> tuple[int, int]:
    def align(value):
        value = max(64, min(4096, int(value or 1024)))
        return value // 16 * 16

    return align(width), align(height)


class model_factory:
    """Load native distilled DiT, Qwen3-VL features, and VAE through MMGP."""

    def __init__(self, model_filename, text_encoder_filename=None, model_def=None,
                 VAE_dtype=torch.float32, dtype=torch.bfloat16, **kwargs):
        from transformers import Qwen3VLConfig, Qwen3VLModel, Qwen3VLProcessor

        self._abort = False
        self.model_def = dict(model_def or {})
        self.device = torch.device("cuda")

        transformer_paths = _expand_numbered_shards(
            _resolve_local_checkpoint(model_filename, "FLUX 3 transformer"), "FLUX 3 transformer"
        )
        text_encoder_folder = self.model_def.get("text_encoder_folder", TEXT_ENCODER_FOLDER)
        text_encoder_config = os.path.join(text_encoder_folder, "config.json")
        config_path = fl.locate_file(text_encoder_config)
        encoder_folder = Path(config_path).parent
        qwen_config = Qwen3VLConfig.from_json_file(config_path)
        if qwen_config.model_type != "qwen3_vl":
            raise ValueError(f"Expected a Qwen3-VL config for FLUX 3, found {qwen_config.model_type!r}.")
        text_config = qwen_config.text_config
        if int(text_config.num_hidden_layers) < max(OUTPUT_LAYERS):
            raise ValueError("The FLUX 3 text encoder requires Qwen3-VL hidden layers through layer 32.")
        if int(text_config.hidden_size) * len(OUTPUT_LAYERS) != FEATURE_WIDTH:
            raise ValueError(
                f"Qwen3-VL hidden size {text_config.hidden_size} does not produce FLUX 3's "
                f"{FEATURE_WIDTH}-wide text features."
            )

        with torch.device("cpu"), init_empty_weights(include_buffers=False):
            self.transformer = DiT(flow_map=True)
            self.text_encoder = Qwen3VLModel(qwen_config)
            self.vae = AutoEncoder()

        # The bare model uses exactly the hidden states the DiT was trained on.
        # Its official checkpoint has a `model.` prefix plus an unused lm_head;
        # shape validation below proves the prefix and head match this Qwen3-VL config.
        configure_encoder_attention(self.text_encoder)
        qwen_paths = self._resolve_text_encoder_paths(text_encoder_filename, text_encoder_folder)
        qwen_paths_for_validation = _expand_numbered_shards(qwen_paths, "FLUX 3 Qwen3-VL")
        _validate_checkpoint_shapes(self.transformer, transformer_paths, "FLUX 3 DiT")
        _validate_checkpoint_shapes(
            self.text_encoder,
            qwen_paths_for_validation,
            "FLUX 3 Qwen3-VL",
            remap=lambda name, shape: _qwen_checkpoint_key(name, shape, qwen_config),
        )

        vae_path = fl.locate_file(VAE_CHECKPOINT)
        _validate_checkpoint_shapes(self.vae, [vae_path], "FLUX 3 VAE")

        # The official reference uses a native FP32 VAE. Lock parameter and
        # buffer modules so MMGP's global float-to-BF16 policy cannot change it.
        _lock_native_vae_fp32(self.vae)
        _load_mmgp_checkpoint(
            self.transformer,
            transformer_paths,
            writable_tensors=False,
            default_dtype=torch.bfloat16,
        )
        _load_mmgp_checkpoint(
            self.text_encoder,
            qwen_paths,
            preprocess_sd=_strip_qwen_checkpoint_prefix,
            writable_tensors=False,
            default_dtype=torch.bfloat16,
        )
        _load_mmgp_checkpoint(
            self.vae,
            [vae_path],
            writable_tensors=False,
            default_dtype=torch.float32,
        )

        self.transformer._model_dtype = torch.bfloat16
        self.text_encoder._model_dtype = torch.bfloat16
        _configure_native_vae_offload(self.vae)
        for module in (self.transformer, self.text_encoder, self.vae):
            module.eval().requires_grad_(False)

        self.processor = Qwen3VLProcessor.from_pretrained(str(encoder_folder), local_files_only=True)
        self.processor.tokenizer.padding_side = "right"

    @staticmethod
    def _resolve_text_encoder_paths(text_encoder_filename, text_encoder_folder: str) -> list[str]:
        if text_encoder_filename is not None:
            return _resolve_local_checkpoint(text_encoder_filename, "FLUX 3 Qwen3-VL")
        paths = []
        for filename in TEXT_ENCODER_SHARDS:
            paths.append(fl.locate_file(os.path.join(text_encoder_folder, filename)))
        return paths

    @property
    def _interrupt(self):
        return self._abort

    @_interrupt.setter
    def _interrupt(self, value):
        self._abort = bool(value)

    @torch.inference_mode()
    def generate(self, input_prompt="", n_prompt="", seed=-1, sampling_steps=DEFAULT_STEPS,
                 width=1024, height=1024, guide_scale=1.0, batch_size=1,
                 input_ref_images=None, original_input_ref_images=None, video_prompt_type="",
                 input_frames=None, input_masks=None, image_start=None, callback=None,
                 set_progress_status=None, sample_solver="default", loras_slists=None,
                 outpainting_dims=None, model_mode=None, **kwargs):
        if _has_active_loras(loras_slists) or _has_active_loras(kwargs.get("activated_loras")):
            raise ValueError("FLUX 3 Image Klein does not support loaded or scheduled LoRAs yet.")
        if input_masks is not None or _active_outpainting(outpainting_dims):
            raise ValueError("FLUX 3 Image Klein supports instruction-based reference editing, not masked inpainting or outpainting.")
        if model_mode not in (None, 0, "0", 1, "1"):
            raise ValueError("FLUX 3 Image Klein supports standard image generation and reference editing only.")
        if str(sample_solver or "default").strip().lower() not in ("", "default", "euler"):
            raise ValueError("FLUX 3 Image Klein uses its native flow-map Euler sampler.")
        try:
            numeric_steps = float(sampling_steps)
            steps = int(numeric_steps)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("FLUX 3 Image Klein sampling steps must be a positive integer.") from error
        if not np.isfinite(numeric_steps) or steps <= 0 or numeric_steps != steps:
            raise ValueError("FLUX 3 Image Klein sampling steps must be a positive integer.")
        if not isinstance(input_prompt, str) or not input_prompt.strip():
            raise ValueError("Enter a description or editing instruction for FLUX 3 Image Klein.")

        width, height = _check_model_size(width, height)
        references = _active_references(video_prompt_type, input_ref_images, original_input_ref_images)
        if len(references) > 1:
            references = [prepare_multi_reference(image) for image in references]
        elif references:
            references = [prepare_reference(references[0])[0]]
        latent_h, latent_w = latent_shape(height, width)
        # The DiT uses the reference's half-scale RoPE coordinates, but VAE
        # decode infers the actual latent canvas from integer row/column IDs.
        decode_ids = latent_ids(latent_h, latent_w, self.device)
        image_ids = scaled_ids(decode_ids).to(device=self.device)
        chosen_seed = int(seed) if seed is not None and int(seed) >= 0 else int(torch.seed())
        batch_size = max(1, int(batch_size or 1))
        schedule = timesteps(steps, DEFAULT_ALPHA)

        def check_cancel(*_):
            if self._abort:
                raise GenerationCancelled()

        handles = [self.transformer.register_forward_pre_hook(check_cancel),
                   self.vae.encoder.register_forward_pre_hook(check_cancel),
                   self.vae.decoder.register_forward_pre_hook(check_cancel)]
        for module in (*getattr(self.text_encoder.visual, "blocks", ()),
                       *getattr(self.text_encoder.language_model, "layers", ())):
            handles.append(module.register_forward_pre_hook(check_cancel))

        try:
            check_cancel()
            _report_batch_start(callback, batch_size, steps)
            if callable(set_progress_status):
                set_progress_status("Encoding Prompt")
            text = encode_prompt(self.text_encoder, self.processor, input_prompt, self.device, check_cancel)
            check_cancel()

            cond_tokens, cond_ids = [], []
            for index, reference in enumerate(references):
                check_cancel()
                if callable(set_progress_status):
                    set_progress_status(f"Encoding Reference {index + 1}/{len(references)}")
                pixels = to_tensor(reference).to(device=self.device, dtype=torch.float32)
                tokens, ids = self.vae.encode(pixels)
                ids = ids.float()
                ids[:, 1:3] *= 0.5
                ids[:, 3] = -(1 + index)
                cond_tokens.append(tokens.to(device=self.device, dtype=torch.bfloat16))
                cond_ids.append(ids.to(device=self.device))
                check_cancel()
            condition = (torch.cat(cond_tokens), torch.cat(cond_ids)) if cond_tokens else None

            outputs = []
            if callable(set_progress_status):
                set_progress_status("Denoising")
            for batch_index in range(batch_size):
                check_cancel()
                generator = torch.Generator(device="cpu").manual_seed(chosen_seed + batch_index)
                noise = torch.randn((latent_h * latent_w, 128), generator=generator)
                # Match the reference's BF16-quantized noise and FP32 Euler state.
                noise = noise.to(torch.bfloat16).float().to(self.device)
                image = (noise, image_ids)
                packed = pack([[text, condition, image]])
                tokens = denoise_flow_map(
                    self.transformer, packed, schedule,
                    on_step=_make_step_callback(callback, batch_index, steps, check_cancel),
                    check_cancel=check_cancel,
                )
                check_cancel()
                if callable(set_progress_status):
                    set_progress_status("VAE Decoding")
                decoded = self.vae.decode(tokens, decode_ids).float()
                if not torch.isfinite(decoded).all():
                    raise RuntimeError(
                        "FLUX 3 Image Klein produced non-finite pixels in its native FP32 VAE."
                    )
                decoded = decoded.clamp(-1, 1)
                check_cancel()
                outputs.append(decoded.cpu())
            # Maestro expects C x B x H x W, with normalized RGB float pixels.
            return torch.stack(outputs, dim=1)
        except GenerationCancelled:
            return None
        finally:
            for handle in handles:
                handle.remove()
