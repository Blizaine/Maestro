"""Maestro registration for the unified Qwen Image 2.1 architecture."""

import os

from shared.utils.hf import build_hf_url


MODEL_TYPE = "qwen_image_21_7B"
REPO = "DeepBeepMeep/Qwen_image_2"
ENCODER_FOLDER = "Qwen3-VL-8B-Instruct"
# Complete Transformers 4.57-compatible processor export before the upstream
# folder was removed and its legacy tokenizer files renamed in Ideogram4.
PROCESSOR_REVISION = "55ab7995df218d8f759c6c283ef4c3be66111345"


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "image_outputs": True,
            "dtype": "bf16",
            "guidance_max_phases": 1,
            "sample_solvers": [("Euler", "default")],
            "compile": False,
            "fit_into_canvas_image_refs": 0,
            "vae_block_size": 32,
            "resolutions_categories": ["<=2k"],
            "profiles_dir": ["qwen21"],
            "no_background_removal": True,
            "max_image_refs": 10,
            "preserve_image_ref_alpha": True,
            "image_ref_choices": {
                "choices": [("None", ""), ("Reference images", "I")],
                "letters_filter": "I",
                "default": "I",
            },
            "at_least_one_image_ref_needed": False,
            "text_encoder_folder": ENCODER_FOLDER,
            "text_encoder_URLs": [
                build_hf_url("DeepBeepMeep/Ideogram4", ENCODER_FOLDER, "Qwen3-VL-8B-Instruct_bf16.safetensors"),
                build_hf_url("DeepBeepMeep/Ideogram4", ENCODER_FOLDER, "Qwen3-VL-8B-Instruct_int8_convrot.safetensors"),
            ],
        }

    @staticmethod
    def query_supported_types():
        return [MODEL_TYPE]

    @staticmethod
    def query_model_family():
        return "qwen"

    @staticmethod
    def query_family_infos():
        return {"qwen": (110, "Qwen")}

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        # The old 20B adapters have different shapes and must not be offered.
        return getattr(args, "lora_dir_qwen21", None) or os.path.join(lora_root, "qwen21")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        # Pin only processor metadata; existing local files and model-weight
        # downloads keep their paths and normal caching behavior.
        return [{
            "repoId": REPO,
            "sourceFolderList": ["qwen_image_21"],
            "fileList": [["qwen_image_21_vae.safetensors"]],
        }, {
            "repoId": REPO,
            "revision": PROCESSOR_REVISION,
            "sourceFolderList": [ENCODER_FOLDER],
            "fileList": [
                ["added_tokens.json", "chat_template.jinja", "config.json", "merges.txt",
                 "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
                 "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json"],
            ],
        }]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, *,
                   text_encoder_filename=None, VAE_dtype=None, save_quantized=False, **kwargs):
        from .runtime import model_factory
        runtime = model_factory(model_filename, text_encoder_filename, model_def=model_def,
                                VAE_dtype=VAE_dtype, save_quantized=save_quantized, model_type=model_type)
        return runtime, {"transformer": runtime.transformer, "text_encoder": runtime.text_encoder,
                         "vae": runtime.vae}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(image_mode=1, video_prompt_type="I", batch_size=1,
                           num_inference_steps=40, guidance_scale=1.0, sample_solver="default",
                           resolution="1024x1024", remove_background_images_ref=0)

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        ui_defaults.setdefault("image_mode", 1)

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if len(inputs.get("image_refs") or []) > 10:
            return "Qwen Image 2.1 supports up to 10 reference images."
