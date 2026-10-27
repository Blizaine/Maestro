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
EDIT_ASSET_REVISION = "66a563156a744feba3806b3528b560bdf6f59fb6"
VAE_ENCODER_REPO = "DeepBeepMeep/Flux2"
VAE_ENCODER_REVISION = "c976cb238dbf2bee48f0bd65eeb14c039cf45e03"
VAE_ENCODER_FILE = "flux2_vae.safetensors"
HQ_VISION_FOLDER = "moss_image_hq_vision"
HQ_VISION_FILE = "Qwen3.5-9B-vision_bf16.safetensors"
HQ_VISION_REPO = "DeepBeepMeep/Wan2.1"
HQ_VISION_REVISION = "a4c0cc35033229072cb6d5a92952c3d51716837f"
HQ_VISION_SOURCE_FOLDER = "Qwen3_5_9B_Abliterated"
HQ_CONFIG_REPO = "Qwen/Qwen3.5-9B"
HQ_CONFIG_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
HQ_CONFIG_FILES = ("config.json", "preprocessor_config.json")


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

        model_def = model_def or {}
        required_assets = [
            *(os.path.join(TEXT_FOLDER, name) for name in TOKENIZER_FILES),
            os.path.join("moss_image", VAE_FILE),
        ]
        definition = {
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
        if model_def.get("moss_edit", False):
            definition.update(_editing_definition(model_def))
        return definition

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        files = [{
            "repoId": REPO,
            "revision": REVISION,
            "sourceFolderList": [TEXT_FOLDER, "moss_image"],
            "targetFolderList": [None, None],
            "fileList": [list(TOKENIZER_FILES), [VAE_FILE]],
        }]
        if (model_def or {}).get("moss_edit", False):
            files.append({
                "repoId": VAE_ENCODER_REPO,
                "revision": VAE_ENCODER_REVISION,
                "sourceFolderList": [""],
                "fileList": [[VAE_ENCODER_FILE]],
            })
            from .edit_runtime import is_hq_edit_variant

            if is_hq_edit_variant((model_def or {}).get("moss_edit_variant")):
                files.extend([{
                    "repoId": HQ_VISION_REPO,
                    "revision": HQ_VISION_REVISION,
                    "sourceFolderList": [HQ_VISION_SOURCE_FOLDER],
                    "targetFolderList": [HQ_VISION_FOLDER],
                    "fileList": [[HQ_VISION_FILE]],
                }, {
                    "repoId": HQ_CONFIG_REPO,
                    "revision": HQ_CONFIG_REVISION,
                    "sourceFolderList": [""],
                    "targetFolderList": [HQ_VISION_FOLDER],
                    "fileList": [list(HQ_CONFIG_FILES)],
                }])
        return files

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
        components = {
            "transformer": pipeline.transformer,
            "text_encoder": pipeline.text_encoder,
            "vae": pipeline.vae,
        }
        if getattr(pipeline, "vision_encoder", None) is not None:
            components["vision_encoder"] = pipeline.vision_encoder
        return pipeline, components

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        model_def = model_def or {}
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
        if model_def.get("moss_edit", False):
            ui_defaults.update({
                "guidance_scale": 3 if str(model_def.get("moss_edit_variant", "")).startswith("hq_") else 4,
                "video_prompt_type": "I",
                "remove_background_images_ref": 0,
                "custom_settings": {"moss_image_kv_cache": "Disabled"},
            })

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        from .edit_runtime import canonicalize_edit_reference_selector
        from .pipeline import _reference_images_from_inputs, validate_generation_inputs

        model_def = model_def or {}
        if isinstance(inputs, dict) and model_def.get("moss_edit", False):
            inputs["video_prompt_type"] = canonicalize_edit_reference_selector(
                inputs.get("video_prompt_type"),
                _reference_images_from_inputs(inputs),
                editing=True,
                image_mode=inputs.get("image_mode", inputs.get("model_mode")),
            )

        return validate_generation_inputs(inputs, model_def=model_def)

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("flux", sub_family="flux2")


def _editing_definition(model_def):
    """Native ordered-reference edits are separate from layout generation."""
    from .prompts import EDITING

    from .edit_runtime import is_hq_edit_variant

    hq = is_hq_edit_variant(model_def.get("moss_edit_variant"))
    definition = {
        "moss_edit": True,
        "moss_image_prompt_helper": False,
        "max_image_refs": 4,
        "preserve_image_ref_native_size": True,
        "at_least_one_image_ref_needed": True,
        "inpaint_support": False,
        "image_ref_choices": {"choices": [("Ordered reference images (1–4)", "I")], "letters_filter": "I", "default": "I"},
        "required_model_assets": [
            *(os.path.join(TEXT_FOLDER, name) for name in TOKENIZER_FILES),
            os.path.join("moss_image", VAE_FILE), VAE_ENCODER_FILE,
        ],
        "moss_vae_encoder_file": VAE_ENCODER_FILE,
        "moss_edit_reference_size": 1024,
        "custom_settings": [{
            "id": "moss_image_kv_cache", "name": "Reference KV cache", "label": "Reference KV cache",
            "type": "dropdown", "default": "Disabled",
            "choices": [
                ("Disabled — recompute references", "Disabled"),
                ("Enabled — full precision in VRAM", "Enabled"),
                ("INT8 — compressed in VRAM", "INT8"),
                ("INT8 in RAM — lower VRAM, transfer overhead", "INT8_RAM"),
            ],
            "info": "Reuse reference keys and values within this generation. More or larger references increase memory. INT8 changes numerical precision; RAM streaming can be slower. Memory checks or step-varying LoRAs can disable caching; the runtime log records the actual path.",
        }],
        "text_prompt_enhancer_max_tokens": 768,
        "image_prompt_enhancer_max_tokens": 768,
        "text_prompt_enhancer_instructions": EDITING,
        "image_prompt_enhancer_instructions": EDITING,
        "text_prompt_enhancer_instructions1": EDITING,
        "text_prompt_enhancer_max_tokens1": 768,
        "prompt_enhancer_def": {
            "selection": ["T", "TI"],
            "labels": {"T": "An Image Editing Instruction", "TI": "An Image Edit using {image_inputs}"},
            "default": "",
        },
        "infos": (
            ("MOSS Edit HQ v5 is an unofficial community editor on MOSS SFT. Start with one source and a bounded edit, 50 steps, CFG 3, shift 5. It uses a separate Qwen vision encoder with a 512-by-512 pixel budget, plus VAE reference features fitted within 1024 pixels per side. " if hq else
             "MOSS Edit SFT uses the original community editing adapter on MOSS SFT. Start at 50 steps, CFG 4, shift 5. References are fitted within 1024 pixels per side. ")
            + "Provide 1–4 images in order, describe what to change and what to preserve, and explain each image's role. The selected canvas sets output dimensions. Surrounding details may change; there is no mask or exact pixel-preservation workflow. Optional reference caching uses extra memory and is disabled by default. Reduce canvas size and batch size for limited VRAM."
        ),
        "prompt_infos": "Use concise edit instructions, for example: Change only the jacket to dark blue; preserve the person's face, pose, background and lighting. For multiple images, say image 1, image 2, etc. in upload order and specify each role. Quote exact lettering. These are natural-language instructions, not special image tokens or layout boxes.",
        "deepy_infos": "Native image-reference editing: image_mode=1, video_prompt_type=I, 1–4 ordered image_refs; output dimensions follow the selected canvas. No masks, inpainting, outpainting, audio or video. 50 Euler steps and shift 5; CFG 3 for HQ v5 or 4 for Edit SFT. Optional custom_settings.moss_image_kv_cache: Disabled, Enabled, INT8, INT8_RAM. Caches last only this request; HQ guidance branches remain independent. Cache fallback is logged. Unchanged pixels are not guaranteed.",
    }
    if hq:
        definition.update({
            "moss_edit_vision_folder": HQ_VISION_FOLDER,
            "moss_edit_vision_file": os.path.join(HQ_VISION_SOURCE_FOLDER, HQ_VISION_FILE),
            "moss_edit_vision_pixels": 512 * 512,
        })
        definition["required_model_assets"].extend(
            os.path.join(HQ_VISION_FOLDER, name) for name in HQ_CONFIG_FILES
        )
        definition["required_model_assets"].append(
            os.path.join(HQ_VISION_FOLDER, HQ_VISION_SOURCE_FOLDER, HQ_VISION_FILE)
        )
    return definition
