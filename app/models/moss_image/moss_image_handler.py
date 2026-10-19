# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Maestro model registration for the native MOSS-Image 5B pipeline."""

import os

import torch

MODEL_TYPE = "moss_image"
REPO = "DeepBeepMeep/Moss"
REVISION = "0a508e986e920c8051b99df98d27329742fc3d0e"
TEXT_FOLDER = "Qwen3.5-9B"
TEXT_ENCODER_FILES = (
    "qwen3_5_9b_text_bf16.safetensors",
    "qwen3_5_9b_text_int8_convrot.safetensors",
)
TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
VAE_FILE = "flux2_small_decoder_bf16.safetensors"


def _hf_url(folder, filename):
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{folder}/{filename}"


class family_handler:
    @staticmethod
    def query_supported_types():
        return [MODEL_TYPE]

    @staticmethod
    def query_model_family():
        return MODEL_TYPE

    @staticmethod
    def query_family_infos():
        return {MODEL_TYPE: (1161, "MOSS Image")}

    @staticmethod
    def query_family_maps():
        return {}, {MODEL_TYPE: [MODEL_TYPE]}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        from .prompts import CLASSIC, STRUCTURED

        required_assets = [
            *(os.path.join(TEXT_FOLDER, name) for name in TOKENIZER_FILES),
            os.path.join("moss_image", VAE_FILE),
        ]
        return {
            "image_outputs": True,
            "device_explicit": True,
            "group": MODEL_TYPE,
            "profiles_dir": [MODEL_TYPE],
            "inference_steps": True,
            "compile": False,
            "auto_quantize": False,
            "guidance_max_phases": 1,
            "embedded_guidance": False,
            "no_negative_prompt": False,
            "no_background_removal": True,
            "vae_block_size": 16,
            "fit_into_canvas_image_refs": 0,
            "max_image_refs": 0,
            "at_least_one_image_ref_needed": False,
            "flow_shift": True,
            "sample_solvers": [("Euler", "default")],
            "skip_prompt_template": True,
            "preserve_empty_prompt_lines": True,
            "moss_image_prompt_helper": True,
            "text_encoder_folder": TEXT_FOLDER,
            "text_encoder_URLs": [_hf_url(TEXT_FOLDER, name) for name in TEXT_ENCODER_FILES],
            "required_model_assets": required_assets,
            "required_model_asset_groups": [[
                os.path.join(TEXT_FOLDER, name) for name in TEXT_ENCODER_FILES
            ]],
            "text_prompt_enhancer_instructions": CLASSIC,
            "text_prompt_enhancer_max_tokens": 2048,
            "text_prompt_enhancer_instructions1": STRUCTURED,
            "text_prompt_enhancer_max_tokens1": 4096,
            "prompt_enhancer_def": {
                "selection": ["T", "T1"],
                "labels": {"T": "A Detailed Image Caption", "T1": "An Image Layout with Bounding Boxes"},
                "default": "",
            },
            "infos": (
                "MOSS Image 5B generates images from natural language, tags, or JSON layouts. "
                "Base and SFT use the same architecture. Defaults are 50 steps, CFG 3, and flow shift 5. "
                "Use dimensions divisible by 16 and keep the canvas at or below 4 megapixels; "
                "1–2 megapixels is recommended. Batch generation uses sequential seeds and needs more memory. "
                "This model accepts text prompts only; it does not edit reference images or masks."
            ),
            "deepy_infos": (
                "Text-to-image only. Defaults: 50 steps, guidance 3, flow shift 5, 1024x1024, batch 1. "
                "Dimensions must be divisible by 16 and the total canvas must not exceed 4 megapixels. "
                "Batch size must be positive; memory use grows with batch size. Negative-branch layer 22 skipping is built in."
            ),
            "prompt_infos": (
                "Describe the subject, framing, arrangement, lighting, materials, style, and exact visible wording. "
                "For layout prompts, use summary.caption and items with label, caption, optional text, and "
                "bbox=[y_min,x_min,y_max,x_max]. Set bbox_coord_space to abs_centralized and bbox_format to yxyx. "
                "Coordinates are integer pixel corners measured from the canvas center; X increases right and Y down. "
                "Keep every box inside the selected dimensions. The model has no reference-image inputs."
            ),
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{
            "repoId": REPO,
            "revision": REVISION,
            "sourceFolderList": [TEXT_FOLDER, "moss_image"],
            "targetFolderList": [None, None],
            "fileList": [list(TOKENIZER_FILES), [VAE_FILE]],
        }]

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        custom = getattr(args, "lora_dir_moss_image", None) if args is not None else None
        return custom or os.path.join(lora_root, MODEL_TYPE)

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
        text_encoder_filename=None,
        save_quantized=False,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        from .pipeline import load_pipeline

        pipeline = load_pipeline(
            model_filename,
            text_encoder_filename,
            model_def or {},
            model_type or base_model_type or MODEL_TYPE,
            save_quantized,
            dtype,
        )
        return pipeline, {
            "transformer": pipeline.transformer,
            "text_encoder": pipeline.text_encoder,
            "vae": pipeline.vae,
        }

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "image_mode": 1,
            "resolution": "1024x1024",
            "batch_size": 1,
            "num_inference_steps": 50,
            "guidance_scale": 3,
            "flow_shift": 5,
            "prompt_enhancer": "",
            "video_prompt_type": "",
            "multi_prompts_gen_type": "FG",
        })

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        from .pipeline import validate_generation_inputs

        return validate_generation_inputs(inputs)

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("flux", sub_family="flux2")
