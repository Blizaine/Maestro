"""Qwen Image 2.1 adapter for Maestro's MMGP generation lifecycle."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from transformers import Qwen3VLConfig, Qwen3VLModel, Qwen3VLProcessor

from mmgp import offload
from shared.utils import files_locator as fl

from .autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
from .pipeline_qwenimage21 import QwenImage21Pipeline
from .transformer_qwenimage21 import QwenImage21Transformer2DModel


CONFIGS = Path(__file__).parent / "configs"


def _config(name):
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def _pil(image):
    if isinstance(image, Image.Image):
        return image
    if torch.is_tensor(image):
        from shared.utils.utils import convert_tensor_to_image
        return convert_tensor_to_image(image)
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return source.copy()
    return Image.fromarray(np.asarray(image))


class GenerationCancelled(Exception):
    pass


class MaestroQwenImage21Pipeline(QwenImage21Pipeline):
    @property
    def _execution_device(self):
        # MMGP, rather than Accelerate hooks, owns component placement.
        return self.execution_device


class model_factory:
    def __init__(self, model_filename, text_encoder_filename, model_def=None,
                 VAE_dtype=None, save_quantized=False, model_type=None):
        self._abort = False
        self.device = torch.device("cuda")
        filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        # Keep non-persistent RoPE buffers real, on CPU. They are not present
        # in the safetensors checkpoints and cannot be materialized from meta.
        with torch.device("cpu"), init_empty_weights(include_buffers=False):
            self.transformer = QwenImage21Transformer2DModel.from_config(_config("transformer_config.json"))
            self.text_encoder = Qwen3VLModel(Qwen3VLConfig.from_dict(_config("text_encoder_config.json")))
            self.vae = AutoencoderKLQwenImage21.from_config(_config("vae_config.json"))
        offload.load_model_data(self.transformer, filename, writable_tensors=False, default_dtype=torch.bfloat16)
        offload.load_model_data(self.text_encoder, text_encoder_filename, writable_tensors=False,
                               default_dtype=torch.bfloat16)
        self.transformer._model_dtype = torch.bfloat16
        self.text_encoder._model_dtype = torch.bfloat16
        # The native VAE is FP32. BF16 is the supported low-memory alternative;
        # never downcast its convolutions to FP16, which can overflow.
        vae_dtype = torch.float32 if VAE_dtype == torch.float32 else torch.bfloat16
        offload.load_model_data(self.vae, fl.locate_file("qwen_image_21/qwen_image_21_vae.safetensors"),
                               writable_tensors=False, default_dtype=vae_dtype)
        self.vae._model_dtype = vae_dtype
        self.vae._offload_hooks = ["encode", "decode"]
        for module in (self.transformer, self.text_encoder, self.vae):
            module.eval().requires_grad_(False)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.transformer, model_type, filename, torch.bfloat16,
                                 str(CONFIGS / "transformer_config.json"))
        processor_dir = Path(fl.locate_file("Qwen3-VL-8B-Instruct/tokenizer_config.json")).parent
        self.processor = Qwen3VLProcessor.from_pretrained(str(processor_dir), local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.pipeline = MaestroQwenImage21Pipeline(
            FlowMatchEulerDiscreteScheduler.from_config(_config("scheduler_scheduler_config.json")),
            self.vae, self.text_encoder, self.processor, self.transformer,
        )
        self.pipeline.execution_device = self.device
        self.pipeline.set_progress_bar_config(desc="Qwen Image 2.1")

    def finalize_loras(self):
        from shared.qtypes.int8_convrot import install_native_lora_forwards
        install_native_lora_forwards(self.transformer)

    @property
    def _interrupt(self):
        return self._abort

    @_interrupt.setter
    def _interrupt(self, value):
        self._abort = bool(value)

    @torch.inference_mode()
    def generate(self, input_prompt="", n_prompt="", seed=-1, sampling_steps=40,
                 width=1024, height=1024, guide_scale=1.0, batch_size=1,
                 input_ref_images=None, original_input_ref_images=None, video_prompt_type="",
                 input_frames=None, image_start=None, VAE_tile_size=None, loras_slists=None,
                 callback=None, set_progress_status=None, **kwargs):
        references = []
        if image_start is not None:
            references.append(_pil(image_start))
        if input_frames is not None:
            references.append(_pil(input_frames))
        if "I" in video_prompt_type:
            refs = original_input_ref_images if original_input_ref_images else input_ref_images
            references.extend(_pil(image) for image in (refs or []))
        if len(references) > 10:
            raise ValueError("Qwen Image 2.1 supports up to 10 reference images.")
        if not input_prompt.strip():
            raise ValueError("Enter a description or editing instruction for Qwen Image 2.1.")
        width, height = max(32, int(width) // 32 * 32), max(32, int(height) // 32 * 32)
        # A single full-resolution RGBA VAE activation can dominate memory.
        # Tile automatically and decode each batch item separately upstream.
        tile = 384
        if isinstance(VAE_tile_size, int) and VAE_tile_size > 0:
            tile = max(128, VAE_tile_size // 32 * 32)
        elif isinstance(VAE_tile_size, (tuple, list)) and len(VAE_tile_size) > 1 and VAE_tile_size[1] > 0:
            tile = max(128, int(VAE_tile_size[1]) // 32 * 32)
        self.vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                               tile_sample_stride_height=tile - 64, tile_sample_stride_width=tile - 64)
        self.vae.enable_slicing()
        if loras_slists is not None:
            from shared.utils.loras_mutipliers import update_loras_slists
            update_loras_slists(self.transformer, loras_slists, sampling_steps)
            offload.set_step_no_for_lora(self.transformer, 0)

        def check_cancel(*_):
            if self._abort:
                raise GenerationCancelled()

        def step_end(pipe, step, timestep, tensors):
            check_cancel()
            if loras_slists is not None:
                offload.set_step_no_for_lora(self.transformer, min(step + 1, sampling_steps - 1))
            if callback is not None:
                callback(step, None, False)
            return tensors

        def decoding(*_):
            check_cancel()
            if callable(set_progress_status):
                set_progress_status("VAE Decoding")

        handles = [self.transformer.register_forward_pre_hook(check_cancel),
                   self.vae.decoder.register_forward_pre_hook(decoding)]
        try:
            check_cancel()
            if callable(set_progress_status):
                set_progress_status("Encoding Prompt")
            chosen_seed = int(seed) if seed is not None and seed >= 0 else torch.seed()
            images = self.pipeline(
                prompt=input_prompt, image=references or None,
                negative_prompt=(n_prompt or " ") if guide_scale > 1 else None,
                true_cfg_scale=float(guide_scale), width=width, height=height,
                output_resolution=min(1024, max(256, int(math.sqrt(width * height)))),
                num_inference_steps=int(sampling_steps), num_images_per_prompt=max(1, int(batch_size)),
                generator=torch.Generator(device=self.device).manual_seed(chosen_seed),
                callback_on_step_end=step_end, callback_on_step_end_tensor_inputs=[],
                # Step-varying adapter weights would invalidate the prefix KV
                # cache. Keep the ordinary cache for jobs without adapters.
                use_kv_cache=not bool(loras_slists and loras_slists.get("phase1")),
                output_type="pt",
            ).images
            check_cancel()
            # Preserve four channels through Maestro's PNG saver. Quantize on
            # CPU so neither a float32 conversion nor a second full image lives
            # on the GPU during saving.
            images = images.cpu().float()
            if not torch.isfinite(images).all():
                raise RuntimeError("Qwen Image 2.1 produced non-finite pixels. Try the 32-bit VAE setting.")
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            return images.transpose(0, 1)
        except GenerationCancelled:
            return None
        finally:
            for handle in handles:
                handle.remove()
            self.vae.clear_cache()
