"""Qwen Image 2.1 adapter for Maestro's MMGP generation lifecycle."""

import itertools
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
from .memory import configure_encoder_attention
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


def _has_scheduled_lora_weights(loras_slists):
    if not isinstance(loras_slists, dict):
        return False
    return any(
        isinstance(value, (list, tuple))
        for phase in ("phase1", "phase2", "phase3")
        for value in (loras_slists.get(phase) or [])
    )


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
        configure_encoder_attention(self.text_encoder)
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
        self.vae._offload_hooks = ["encode", "decode", "decode_to_cpu_uint8"]
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
                 width=1024, height=1024, guide_scale=4.0, batch_size=1,
                 input_ref_images=None, original_input_ref_images=None, video_prompt_type="",
                 input_frames=None, input_masks=None, image_start=None, VAE_tile_size=None, loras_slists=None,
                 sample_solver="default", denoising_strength=1.0, masking_strength=1.0,
                 model_mode=0, outpainting_dims=None, custom_settings=None,
                 callback=None, set_progress_status=None, **kwargs):
        selector = str(video_prompt_type or "")
        image_mode = kwargs.get("image_mode", 1)
        if isinstance(image_mode, str):
            try:
                image_mode = int(image_mode)
            except ValueError:
                image_mode = 1
        try:
            active_outpaint = any(float(value) > 0 for value in (outpainting_dims or ()))
        except (TypeError, ValueError):
            active_outpaint = False
        active_control = "V" in selector or (
            (input_masks is not None or active_outpaint) and image_mode == 2
        )
        references = []

        control_source = input_frames if active_control else None
        if control_source is not None:
            references.append(_pil(control_source))
        if image_start is not None:
            references.append(_pil(image_start))
        if "I" in selector:
            refs = original_input_ref_images if original_input_ref_images else input_ref_images
            references.extend(_pil(image) for image in (refs or []))
        if len(references) > 10:
            raise ValueError("Qwen Image 2.1 supports up to 10 reference images.")
        if not input_prompt.strip():
            raise ValueError("Enter a description or editing instruction for Qwen Image 2.1.")
        width, height = max(32, int(width) // 32 * 32), max(32, int(height) // 32 * 32)
        # Match WanGP's VRAM-sensitive Qwen tile policy. A 3090 uses 1024px
        # tiles with 25% overlap; smaller GPUs get progressively smaller tiles.
        tile_choice = VAE_tile_size
        if tile_choice is None:
            memory_mb = (torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1048576
                         if torch.cuda.is_available() else 0)
            chooser = getattr(self.vae, "get_VAE_tile_size", None)
            tile_choice = chooser(0, memory_mb, False) if chooser is not None else (True, 384)
        if isinstance(tile_choice, dict):
            tile_enabled = bool(tile_choice.get("enabled", True))
            tile = int(tile_choice.get("tile_sample_min_size", tile_choice.get("tile_size", 1024)))
        elif isinstance(tile_choice, bool):
            tile_enabled, tile = tile_choice, 256
        elif isinstance(tile_choice, (tuple, list)):
            tile_enabled = bool(tile_choice[0]) if tile_choice else True
            tile = int(tile_choice[1]) if len(tile_choice) > 1 else 1024
        elif isinstance(tile_choice, int):
            tile_enabled, tile = tile_choice > 0, tile_choice or 1024
        else:
            tile_enabled, tile = True, 1024
        if tile_enabled:
            tile = max(128, tile // 32 * 32)
            stride = max(96, (tile * 3 // 4) // 32 * 32)
            self.vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                                   tile_sample_stride_height=stride, tile_sample_stride_width=stride)
        else:
            self.vae.use_tiling = False
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
        # MMGP can spend a long time in the encoder before the first denoising
        # callback. Let cancellation take effect at each vision/text block.
        for block in itertools.chain(self.text_encoder.visual.blocks, self.text_encoder.language_model.layers):
            handles.append(block.register_forward_pre_hook(check_cancel))
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
                sample_solver=sample_solver, mask_image=input_masks,
                inpaint_image=(references[0] if input_masks is not None and references else None),
                denoising_strength=denoising_strength, masking_strength=masking_strength,
                model_mode=model_mode, outpainting_dims=outpainting_dims,
                # The model-specific cache flag is opt-in and remains subject
                # to the pipeline's conservative per-reference VRAM budget.
                # Scheduled/fused adapter changes invalidate cached prefix K/V.
                use_kv_cache=(custom_settings or {}).get("qwen21_kv_cache", "Disabled") == "Enabled"
                and not _has_scheduled_lora_weights(loras_slists),
                output_type="uint8",
            ).images
            check_cancel()
            # Preserve four channels through Maestro's PNG saver. Quantize on
            # CPU so neither a float32 conversion nor a second full image lives
            # on the GPU during saving.
            if images.dtype != torch.uint8:
                images = images.cpu().float()
                if not torch.isfinite(images).all():
                    raise RuntimeError("Qwen Image 2.1 produced non-finite pixels. Try the 32-bit VAE setting.")
                images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            else:
                images = images.cpu()
            return images.transpose(0, 1)
        except GenerationCancelled:
            return None
        finally:
            for handle in handles:
                handle.remove()
            self.vae.clear_cache()
