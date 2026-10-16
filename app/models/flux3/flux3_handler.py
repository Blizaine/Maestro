"""FLUX 3 Image Klein registration and pinned, on-demand component downloads."""

import os


MODEL_TYPE = "flux3_klein"
REPO = "DeepBeepMeep/Flux3"
REVISION = "1e7c96984937fef03326e3b297fb44a28c30736d"
COMPONENT_FOLDER = "flux3_klein"
TEXT_FOLDER = "Qwen3-VL-4B-Instruct"
ENCODER_FILES = [
    "config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "LICENSE",
]
ROOT_FILES = ["flux2-vae.safetensors"]


class family_handler:
    @staticmethod
    def query_supported_types():
        return [MODEL_TYPE, "flux3_klein_base"]

    @staticmethod
    def query_model_family():
        return "flux"

    @staticmethod
    def query_family_infos():
        return {"flux3": (102, "Flux 3")}

    @staticmethod
    def query_family_maps():
        # Flux 1/2 checkpoints and adapters have different tensor layouts.
        return {}, {}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        distilled = base_model_type == MODEL_TYPE
        return {
            "group": "flux3", "image_outputs": True, "device_explicit": True,
            "dtype": "bf16", "auto_quantize": False, "compile": False,
            "guidance_max_phases": 0 if distilled else 1, "lock_guidance_scale": distilled,
            "no_negative_prompt": distilled, "loras_disabled": True,
            "flux3_distilled": distilled, "flux3_prompt_helper": True,
            "inference_steps": True,
            "sample_solvers": [("Flow-map Euler", "default")],
            "vae_block_size": 16, "resolutions_categories": ["<=2k"],
            "fit_into_canvas_image_refs": 0, "no_background_removal": True,
            "preserve_image_ref_native_size": True, "max_image_refs": 10,
            "image_ref_choices": {
                "choices": [("None", ""), ("Ordered reference images / image editing", "I"), ("Main reference sets output shape", "KI")],
                "letters_filter": "KI", "default": "I",
            },
            "at_least_one_image_ref_needed": False,
            "text_encoder_folder": f"{COMPONENT_FOLDER}/{TEXT_FOLDER}",
            "text_encoder_URLs": [f"https://huggingface.co/{REPO}/resolve/{REVISION}/{TEXT_FOLDER}/qwen3vl_4b_{fmt}.safetensors" for fmt in ("bf16", "int8_convrot")],
            # Layout guidance is expressed in the prompt, never as a hard mask.
            "inpaint_support": False,
            "profiles_dir": [base_model_type],
            "required_model_assets": [
                f"flux3/{name}" for name in ROOT_FILES
            ] + [f"{COMPONENT_FOLDER}/{TEXT_FOLDER}/{name}" for name in ENCODER_FILES],
            "required_model_asset_groups": [[f"{COMPONENT_FOLDER}/{TEXT_FOLDER}/qwen3vl_4b_{fmt}.safetensors" for fmt in ("bf16", "int8_convrot")]],
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{
            "repoId": REPO, "revision": REVISION,
            "sourceFolderList": ["flux3", TEXT_FOLDER],
            "targetFolderList": [None, COMPONENT_FOLDER],
            "fileList": [ROOT_FILES, ENCODER_FILES],
        }]

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return os.path.join(lora_root, MODEL_TYPE)

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, **kwargs):
        from .runtime import model_factory
        runtime = model_factory(model_filename, model_def=model_def, text_encoder_filename=kwargs.get("text_encoder_filename"),
                                VAE_dtype=kwargs.get("VAE_dtype"))
        return runtime, {"transformer": runtime.transformer,
                         "text_encoder": runtime.text_encoder, "vae": runtime.vae}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        distilled = base_model_type == MODEL_TYPE
        ui_defaults.update(image_mode=1, video_prompt_type="I", batch_size=1,
                           num_inference_steps=16 if distilled else 50, guidance_scale=1.0 if distilled else 3.5,
                           embedded_guidance_scale=1.0, sample_solver="default",
                           resolution="1024x1024", remove_background_images_ref=0)

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if "nvfp4" in str((model_def or {}).get("name", "")).lower():
            from services.hardware_detect import detect_hardware
            if not detect_hardware().get("supports_nvfp4", False):
                return "Flux 3 NVFP4 requires an RTX 50 series GPU and a compatible NVFP4 kernel. Use BF16 or INT8 ConvRot on this GPU."
        if inputs.get("activated_loras"):
            return "FLUX 3 Klein LoRA loading has not yet been verified. Remove the adapters for this model."
        if inputs.get("image_mode", 1) in (2, "2") or inputs.get("image_mask") is not None:
            return "Use Image Generate with reference images to edit with FLUX 3 Klein. Masked inpainting is not supported yet."
        selector = str(inputs.get("video_prompt_type") or "")
        if selector not in ("", "I", "KI"):
            return "FLUX 3 Klein supports text-to-image and reference-image editing. Select Reference images or None."
        refs = inputs.get("image_refs") or []
        if "I" in selector and len(refs if isinstance(refs, (list, tuple)) else [refs]) > 10:
            return "FLUX 3 Klein supports up to 10 reference images."
        if str(inputs.get("sample_solver") or "default") not in ("default", "euler"):
            return "FLUX 3 Klein uses its native flow-map Euler sampler."
        return None
