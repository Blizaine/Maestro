# Adapted from Hy-Nomas (https://github.com/Tencent-Hunyuan/Hy-Nomas, models/hy_nomas/hy_nomas_handler.py), Copyright (C) 2026 Tencent, Apache-2.0.
# Modified for Maestro. Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0.
"""Maestro model registration for the Hy-Nomas session runtime."""

import os

from .camera import PINNED_CAMERA_ATTENTION

MODEL_TYPE = "hy_nomas"
TEXT_FOLDER = "Qwen2.5-VL-7B-Instruct"
ASSET_REPO = "DeepBeepMeep/Worlds"
ASSET_REVISION = "545ad7b2393ad5e606c821fb04e07f8fec151046"
QWEN_ASSET_REPO = "DeepBeepMeep/Qwen_image"
QWEN_ASSET_REVISION = "75c70695c563a235a5d14e37f110d499b3ecbd1c"
DIT_FILE = "hy_nomas_distilled_int8.safetensors"
DECODER_FILE = "hy_nomas/pixelumm_decoder_fp8.safetensors"
TOKENIZER_FILES = ("merges.txt", "tokenizer_config.json", "config.json", "vocab.json", "chat_template.json")
TEXT_ENCODER_FILES = (
    "Qwen2.5-VL-7B-Instruct_bf16.safetensors",
    "Qwen2.5-VL-7B-Instruct_quanto_bf16_int8.safetensors",
)
DIT_LFS_SHA256 = "973fa409f3b009232db3c83feddc89c5326fc3a5672feb791f0989ea8bf83252"
DIT_SIZE_BYTES = 9059126120
DECODER_LFS_SHA256 = "5528323e171afd302cece26b05a8dfea8dc8d89e0501734c732b8ef36e098b12"
DECODER_SIZE_BYTES = 7301500088
TEXT_ENCODER_LFS_SHA256 = {
    "Qwen2.5-VL-7B-Instruct_bf16.safetensors": "09a25316d7d9a34686a44ded496e95af32404f95c2fb7e6ee02f4b53b153695e",
    "Qwen2.5-VL-7B-Instruct_quanto_bf16_int8.safetensors": "18966ca6ba18261ec0465442580ad30510a758ecc0314871a0bbd30032716568",
}
TEXT_ENCODER_SIZE_BYTES = {
    "Qwen2.5-VL-7B-Instruct_bf16.safetensors": 16584431274,
    "Qwen2.5-VL-7B-Instruct_quanto_bf16_int8.safetensors": 9432992295,
}
PIXELUMM_LICENSE_URL = "https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf"


def _world_runtime_notice():
    return (
        "Experimental Worlds uses durable queued jobs for creation, continuation, and discrete moves. Each action is a normal model "
        "generation and is not real-time exploration. Before exposing the workflow, the application must verify the DiT, Qwen text "
        "encoder and tokenizer, and PixelUMM decoder are ready, then preserve the generated video and .wgpworld sidecar."
    )


class family_handler:
    @staticmethod
    def query_supported_types():
        return [MODEL_TYPE]

    @staticmethod
    def query_model_family():
        return MODEL_TYPE

    @staticmethod
    def query_family_infos():
        return {MODEL_TYPE: (1166, "Hy-Nomas")}

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return os.path.join(lora_root, MODEL_TYPE)

    @staticmethod
    def query_model_def(base_model_type, model_def):
        required_assets = [
            *(os.path.join(TEXT_FOLDER, name) for name in TOKENIZER_FILES),
            DECODER_FILE,
        ]
        return {
            "device_explicit": True,
            "world_model": True,
            "custom_guide": {"label": "Saved world", "required": False, "file_types": [".wgpworld"]},
            "runtime_custom_settings": ["world_output", "acknowledge_pixelumm_noncommercial"],
            "supports_latent_preview": False,
            "hy_nomas_camera_attention": dict(PINNED_CAMERA_ATTENTION),
            "dtype": "bf16",
            "accelerated": "native",
            "fps": 24,
            "vae_block_size": 32,
            "frames_minimum": 5,
            "frames_steps": 4,
            "frames_selection_maximum": 1441,
            "sliding_window": False,
            "image_prompt_types_allowed": "",
            "guidance_max_phases": 0,
            "no_negative_prompt": True,
            "skip_prompt_template": True,
            "inference_steps": True,
            "profiles_dir": [MODEL_TYPE],
            "text_encoder_folder": TEXT_FOLDER,
            "text_encoder_URLs": [
                f"https://huggingface.co/{QWEN_ASSET_REPO}/resolve/{QWEN_ASSET_REVISION}/{TEXT_FOLDER}/{name}"
                for name in TEXT_ENCODER_FILES
            ],
            "hy_nomas_asset_provenance": {
                "dit": {"repo_id": ASSET_REPO, "revision": ASSET_REVISION, "path": DIT_FILE,
                        "sha256": DIT_LFS_SHA256, "size_bytes": DIT_SIZE_BYTES},
                "decoder": {"repo_id": ASSET_REPO, "revision": ASSET_REVISION, "path": DECODER_FILE,
                            "sha256": DECODER_LFS_SHA256, "size_bytes": DECODER_SIZE_BYTES},
                "text_encoder": {"repo_id": QWEN_ASSET_REPO, "revision": QWEN_ASSET_REVISION,
                                 "files": {name: {"sha256": TEXT_ENCODER_LFS_SHA256[name],
                                                  "size_bytes": TEXT_ENCODER_SIZE_BYTES[name]}
                                           for name in TEXT_ENCODER_FILES}},
            },
            "required_model_assets": required_assets,
            "required_model_asset_groups": [[
                os.path.join(TEXT_FOLDER, name) for name in TEXT_ENCODER_FILES
            ]],
            "license_name": "Apache-2.0 DiT code and weights; PixelUMM decoder weights are non-commercial only",
            "license_url": PIXELUMM_LICENSE_URL,
            "license_acknowledgement_required": True,
            "license_acknowledgement_key": "acknowledge_pixelumm_noncommercial",
            "license_notice": (
                "World generation and exploration use the PixelUMM fine-tuned decoder. Its weights and derivatives may be used only for "
                "non-commercial research or evaluation under NVIDIA's One-Way Noncommercial License. The runtime blocks model loading "
                "unless the app passes explicit acknowledgement."
            ),
            "runtime_notice": _world_runtime_notice(),
            "infos": (
                "Hy-Nomas generates 24 fps segmented video from text. Experimental Worlds creates, continues, and moves through durable "
                "queued generation jobs; movement is not real time. The versioned .wgpworld sidecar preserves the world for later continuation. "
                "Defaults are 8 steps, guidance 1, and 832x480."
            ),
            "prompt_infos": (
                "Use a natural-language world description or a JSON object with perspective_view, Style, subject_list, and ordered segments. "
                "Segments may carry start_time/end_time, visual_description_long or description_change, contained_actions, and camera. "
                "Bracketed move instructions can follow the description, such as [Forward 2s] or [Event] a fox jumps onto a stump."
            ),
            "deepy_infos": (
                "Hy-Nomas is a 24 fps, 32x spatial and 4x temporal compressed world model. It uses a distilled 8-step DiT and a separate "
                "PixelUMM decoder. The decoder checkpoint is restricted to non-commercial research or evaluation."
            ),
            "deepy_prompt_infos": (
                "Write one JSON world description with a perspective_view, Style, subject_list, and segments. Use timestamps to make the "
                "opening video length explicit. Camera trajectories are optional. Use a saved .wgpworld world for continuations; each move "
                "is a queued generation rather than live camera control."
            ),
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{
            "repoId": QWEN_ASSET_REPO,
            "revision": QWEN_ASSET_REVISION,
            "sourceFolderList": [TEXT_FOLDER],
            "fileList": [list(TOKENIZER_FILES)],
        }, {
            "repoId": ASSET_REPO,
            "revision": ASSET_REVISION,
            "sourceFolderList": ["hy_nomas"],
            "fileList": [[os.path.basename(DECODER_FILE)]],
        }]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, text_encoder_filename=None,
                   save_quantized=False, acknowledge_pixelumm_noncommercial=False, **kwargs):
        from .hy_nomas_main import model_factory
        from mmgp import quant_router

        if acknowledge_pixelumm_noncommercial is not True:
            # Surface the precise restriction before a gated decoder load.
            from .hy_nomas_main import require_pixelumm_acknowledgement
            require_pixelumm_acknowledgement(acknowledge_pixelumm_noncommercial)
        quant_router.register_handler("models.hy_nomas.block_scaled")
        model = model_factory(
            model_filename,
            model_type,
            base_model_type,
            text_encoder_filename,
            save_quantized=save_quantized,
            acknowledge_pixelumm_noncommercial=acknowledge_pixelumm_noncommercial,
            camera_attention_config=model_def.get("hy_nomas_camera_attention"),
        )
        pipe = {"transformer": model.transformer, "text_encoder": model.text_encoder.model, "decoder": model.decoder}
        offload_kwargs = {"pipe": pipe, "coTenantsMap": {"transformer": ["decoder"], "decoder": ["transformer"]}, "budgets": {"decoder": 100}}
        if kwargs.get("profile") in (3, 4):
            offload_kwargs["pinnedMemory"] = ["transformer", "decoder"]
        return model, offload_kwargs

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "video_length": 97,
            "resolution": "832x480",
            "image_prompt_type": "",
            "multi_prompts_gen_type": "FG",
        })

def test_hy_nomas(base_model_type):
    return base_model_type == MODEL_TYPE
