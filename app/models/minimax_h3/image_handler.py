"""Native MiniMax H3 Image registration and shared H3 encoder assets."""

import os

from .minimax_h3_handler import (
    _ASSETS_ROOT as TEXT_ENCODER_FOLDER,
    _OFFICIAL_REPO as TEXT_REPO,
    _OFFICIAL_REVISION as TEXT_REVISION,
    _TEXT_ENCODER as TEXT_ENCODER_FILE,
    _WANGP_TEXT_ENCODER_FOLDER,
    _text_encoder_variants,
)


ARCHITECTURE = "h3_image"
MODEL_FAMILY = "minimax_h3_image"
ASSETS_FOLDER = "minimax_h3_image"
REPO = "Comfy-Org/MiniMax-H3-Image"
REVISION = "5e60435c12302e0fa946b35a309736eb25b9272b"
VAE_FILE = "minimax_h3_image_vae_fp16.safetensors"
STEPS = 6
MAX_REFS = 16
PROCESSOR_FILES = [
    "chat_template.json", "merges.txt", "preprocessor_config.json", "tokenizer.json",
    "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
]

_CANVASES = {
    "1k": ("1K", "Quick drafts at roughly one megapixel.", {
        "1:1": "1024x1024", "4:3": "1184x896", "3:4": "896x1184",
        "16:9": "1376x768", "9:16": "768x1376", "21:9": "1568x672",
    }),
    "2k": ("2K", "About four megapixels. A practical starting point.", {
        "1:1": "2048x2048", "4:3": "2368x1792", "3:4": "1792x2368",
        "16:9": "2752x1536", "9:16": "1536x2752", "21:9": "3136x1344",
    }),
    "4k": ("4K", "Native 16-megapixel generation. Requires more memory and time.", {
        "1:1": "4096x4096", "4:3": "4736x3552", "3:4": "3552x4736",
        "16:9": "5472x3072", "9:16": "3072x5472", "21:9": "6272x2688",
    }),
}


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE]

    @staticmethod
    def query_model_family():
        return MODEL_FAMILY

    @staticmethod
    def query_family_infos():
        return {MODEL_FAMILY: (110, "MiniMax H3 Image")}

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "group": MODEL_FAMILY,
            "image_outputs": True,
            "instruction_image_edit": True,
            "returns_audio": False,
            "device_explicit": True,
            "dtype": "bf16",
            "auto_quantize": False,
            "compile": False,
            "loras_disabled": True,
            "inference_steps": True,
            "lock_inference_steps": True,
            "inference_steps_min": STEPS,
            "inference_steps_max": STEPS,
            "guidance_max_phases": 0,
            "lock_guidance_scale": True,
            "no_negative_prompt": True,
            "sample_solvers": [("MiniMax six-step Euler", "euler")],
            "vae_block_size": 32,
            "resolutions": [
                (f"{label} · {ratio} ({size})", size)
                for label, _hint, values in _CANVASES.values()
                for ratio, size in values.items()
            ],
            "resolution_preset_order": list(_CANVASES),
            "resolution_presets": {
                key: {"label": label, "hint": hint, "values": {**values, "auto": values["1:1"]}}
                for key, (label, hint, values) in _CANVASES.items()
            },
            "default_resolution_preset": "2k",
            "supports_auto_aspect": True,
            "fit_into_canvas_image_refs": 0,
            "preserve_image_ref_native_size": True,
            "no_background_removal": True,
            "max_image_refs": MAX_REFS,
            "image_ref_choices": {
                "choices": [("None", ""), ("Ordered reference images", "I"), ("Match first reference aspect", "KI")],
                "letters_filter": "KI",
                "default": "I",
            },
            "at_least_one_image_ref_needed": False,
            "inpaint_support": False,
            "text_encoder_folder": TEXT_ENCODER_FOLDER,
            "text_encoder_URLs": _text_encoder_variants()["nvfp4_awq"]["URLs"],
            "compatible_text_encoder_paths": {
                TEXT_ENCODER_FILE: [os.path.join(_WANGP_TEXT_ENCODER_FOLDER, TEXT_ENCODER_FILE)],
            },
            "minimax_h3_image_vae_filename": f"{ASSETS_FOLDER}/vae/{VAE_FILE}",
            "required_model_assets": [
                f"{ASSETS_FOLDER}/vae/{VAE_FILE}",
            ],
            "required_model_asset_groups": [
                [f"{TEXT_ENCODER_FOLDER}/{TEXT_ENCODER_FILE}", f"{_WANGP_TEXT_ENCODER_FOLDER}/{TEXT_ENCODER_FILE}"],
                [f"{TEXT_ENCODER_FOLDER}/text_encoder/config.json", f"{TEXT_ENCODER_FOLDER}/config.json"],
                *(
                    [f"{TEXT_ENCODER_FOLDER}/processor/{name}", f"{TEXT_ENCODER_FOLDER}/{name}"]
                    for name in PROCESSOR_FILES
                ),
            ],
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [
            {
                "repoId": REPO, "revision": REVISION,
                "sourceFolderList": ["vae"], "targetFolderList": [ASSETS_FOLDER],
                "fileList": [[VAE_FILE]],
            },
            {
                "repoId": TEXT_REPO, "revision": TEXT_REVISION,
                "sourceFolderList": ["processor", "text_encoder"],
                "targetFolderList": [TEXT_ENCODER_FOLDER, TEXT_ENCODER_FOLDER],
                "fileList": [PROCESSOR_FILES, ["config.json"]],
            },
        ]

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return os.path.join(lora_root, MODEL_FAMILY)

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, **kwargs):
        from .image_runtime import model_factory

        return model_factory(
            model_filename,
            model_def=model_def,
            text_encoder_filename=kwargs.get("text_encoder_filename"),
            VAE_dtype=kwargs.get("VAE_dtype"),
            dtype=kwargs.get("dtype"),
        )

    @staticmethod
    def apply_acceleration_profile(settings, *, model_def=None):
        settings.update(num_inference_steps=STEPS, guidance_scale=1.0, sample_solver="euler")
        return settings

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        family_handler.apply_acceleration_profile(ui_defaults)
        ui_defaults.update(
            image_mode=1, video_prompt_type="I", batch_size=1, resolution="2048x2048",
            remove_background_images_ref=0,
        )

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if str(inputs.get("image_mode", 1)) != "1" or inputs.get("image_mask") is not None:
            return "Use Image Generate with reference images to edit with MiniMax H3 Image."
        if str(inputs.get("video_prompt_type") or "") not in ("", "I", "KI"):
            return "MiniMax H3 Image supports text-to-image and ordered image references."
        refs = inputs.get("image_refs") or []
        if len(refs if isinstance(refs, (list, tuple)) else [refs]) > MAX_REFS:
            return f"MiniMax H3 Image accepts up to {MAX_REFS} reference images."
        return None
