"""FLUX 3 Image Klein registration and pinned, on-demand component downloads."""

import os


MODEL_TYPE = "flux3_klein"
REPO = "black-forest-labs/flux-3-image-klein"
REVISION = "8454017f9ebaada30c998fccd9f67a80c7580f68"
COMPONENT_FOLDER = "flux3_klein"
ENCODER_FILES = [
    "config.json", "generation_config.json", "chat_template.json",
    "merges.txt", "preprocessor_config.json", "tokenizer.json",
    "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
    "model.safetensors.index.json", "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]
ROOT_FILES = ["ae.safetensors", "config.json", "manifest.json", "LICENSE.md", "NOTICE.md"]


class family_handler:
    @staticmethod
    def query_supported_types():
        return [MODEL_TYPE]

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
        return {
            "group": "flux3", "image_outputs": True, "device_explicit": True,
            "dtype": "bf16", "auto_quantize": False, "compile": False,
            "guidance_max_phases": 0, "lock_guidance_scale": True,
            "no_negative_prompt": True, "loras_disabled": True,
            "inference_steps": True,
            "sample_solvers": [("Flow-map Euler", "default")],
            "vae_block_size": 16, "resolutions_categories": ["<=2k"],
            "fit_into_canvas_image_refs": 0, "no_background_removal": True,
            "preserve_image_ref_native_size": True, "max_image_refs": 10,
            "image_ref_choices": {
                "choices": [("None", ""), ("Reference images / image editing", "I")],
                "letters_filter": "I", "default": "I",
            },
            "at_least_one_image_ref_needed": False,
            "text_encoder_folder": f"{COMPONENT_FOLDER}/text_encoder",
            # Masked denoising, layout boxes and other FLUX-family controls
            # require their own verified adapter; do not expose inert inputs.
            "inpaint_support": False,
            "profiles_dir": [MODEL_TYPE],
            "required_model_assets": [
                f"{COMPONENT_FOLDER}/{name}" for name in ROOT_FILES
            ] + [f"{COMPONENT_FOLDER}/text_encoder/{name}" for name in ENCODER_FILES],
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{
            "repoId": REPO, "revision": REVISION,
            "sourceFolderList": ["", "text_encoder"],
            "targetFolderList": [COMPONENT_FOLDER, COMPONENT_FOLDER],
            "fileList": [ROOT_FILES, ENCODER_FILES],
        }]

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return os.path.join(lora_root, MODEL_TYPE)

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, **kwargs):
        from .runtime import model_factory
        runtime = model_factory(model_filename, model_def=model_def,
                                VAE_dtype=kwargs.get("VAE_dtype"))
        return runtime, {"transformer": runtime.transformer,
                         "text_encoder": runtime.text_encoder, "vae": runtime.vae}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(image_mode=1, video_prompt_type="I", batch_size=1,
                           num_inference_steps=16, guidance_scale=1.0,
                           embedded_guidance_scale=1.0, sample_solver="default",
                           resolution="1024x1024", remove_background_images_ref=0)

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if inputs.get("activated_loras"):
            return "FLUX 3 Klein LoRA loading has not yet been verified. Remove the adapters for this model."
        if inputs.get("image_mode", 1) in (2, "2") or inputs.get("image_mask") is not None:
            return "Use Image Generate with reference images to edit with FLUX 3 Klein. Masked inpainting is not supported yet."
        selector = str(inputs.get("video_prompt_type") or "")
        if any(letter not in "I" for letter in selector):
            return "FLUX 3 Klein supports text-to-image and reference-image editing. Select Reference images or None."
        refs = inputs.get("image_refs") or []
        if "I" in selector and len(refs if isinstance(refs, (list, tuple)) else [refs]) > 10:
            return "FLUX 3 Klein supports up to 10 reference images."
        if str(inputs.get("sample_solver") or "default") not in ("default", "euler"):
            return "FLUX 3 Klein uses its native flow-map Euler sampler."
        return None
