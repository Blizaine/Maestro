"""Maestro family handler for MiniMax H3 Base.

This first integration exposes the unified FL2VA checkpoint: text-to-video,
image-to-video, and first/last-frame video with native stereo audio.  The
separate Ref2VA/edit checkpoint is intentionally not advertised until its
reference-video/audio packing path is implemented as well.
"""

from __future__ import annotations

import os

import torch


_MODEL_TYPE = "minimax_h3"
_COMFY_REPO = "Comfy-Org/MiniMax-H3"
_COMFY_REVISION = "0543966fbdce5ba05709a8f2031c94bdba629b4a"
_OFFICIAL_REPO = "MiniMaxAI/MiniMax-H3"
_OFFICIAL_REVISION = "5d9b308a59ab12e67147f191e184baf704185bd1"
_ASSETS_ROOT = "minimax_h3"

_TRANSFORMER = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


def _hf_url(repo_id: str, revision: str, *parts: str) -> str:
    path = "/".join(part.strip("/\\") for part in parts if part)
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{path}"


_RESOLUTIONS = [
    ("1344x768 (16:9 native)", "1344x768"),
    ("768x1344 (9:16 native)", "768x1344"),
    ("1024x768 (4:3 native)", "1024x768"),
    ("768x1024 (3:4 native)", "768x1024"),
    ("768x768 (1:1 native)", "768x768"),
    ("1152x640 (16:9)", "1152x640"),
    ("640x1152 (9:16)", "640x1152"),
    ("960x544 (16:9)", "960x544"),
    ("544x960 (9:16)", "544x960"),
    ("864x480 (16:9 low VRAM)", "864x480"),
    ("480x864 (9:16 low VRAM)", "480x864"),
    ("640x640 (1:1 low VRAM)", "640x640"),
    ("608x352 (16:9 minimum)", "608x352"),
    ("352x608 (9:16 minimum)", "352x608"),
]


class family_handler:
    @staticmethod
    def query_supported_types():
        return [_MODEL_TYPE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "minimax_h3"

    @staticmethod
    def query_family_infos():
        return {"minimax_h3": (55, "MiniMax H3")}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "dtype": "bf16",
            "fps": 24,
            # H3's video VAE accepts only 17*n+5 frames.  124 is the first
            # valid count at or above five seconds; 345 is the last at or
            # below fifteen seconds.
            "frames_minimum": 124,
            "frames_steps": 17,
            "frames_maximum": 345,
            "latent_size": 17,
            "frame_alignment_modulus": 17,
            "frame_alignment_remainder": 5,
            "frame_alignment_mode": "ceil",
            "sliding_window": False,
            "t2v_class": True,
            "i2v_class": True,
            "image_prompt_types_allowed": "TSE",
            "end_frames_always_enabled": True,
            "returns_audio": True,
            "no_negative_prompt": True,
            "guidance_max_phases": 0,
            "visible_phases": 0,
            "compile": False,
            "resolutions": _RESOLUTIONS,
            "profiles_dir": ["minimax_h3"],
            "minimax_h3_assets_root": _ASSETS_ROOT,
            "text_encoder_folder": _ASSETS_ROOT,
            "text_encoder_quantization": "int8",
            "text_encoder_URLs": [
                _hf_url(_COMFY_REPO, _COMFY_REVISION, "text_encoders", _TEXT_ENCODER)
            ],
        }

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-minimax-h3",
            type=str,
            default=None,
            help=(
                "Path to a directory that contains MiniMax H3 LoRAs "
                f"(default: {os.path.join(lora_root, 'minimax_h3')})"
            ),
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_minimax_h3", None) or os.path.join(lora_root, "minimax_h3")

    @staticmethod
    def get_vae_block_size(base_model_type):
        return 32

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        processor_files = [
            "chat_template.json",
            "merges.txt",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ]
        return [
            {
                "repoId": _COMFY_REPO,
                "revision": _COMFY_REVISION,
                "sourceFolderList": ["vae"],
                "targetFolderList": [_ASSETS_ROOT],
                "fileList": [[_VIDEO_VAE, _AUDIO_VAE]],
            },
            {
                "repoId": _OFFICIAL_REPO,
                "revision": _OFFICIAL_REVISION,
                "sourceFolderList": ["processor", "text_encoder"],
                "targetFolderList": [_ASSETS_ROOT, _ASSETS_ROOT],
                "fileList": [processor_files, ["config.json"]],
            },
        ]

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
        dtype=torch.bfloat16,
        text_encoder_filename=None,
        **kwargs,
    ):
        from .minimax_h3_main import MiniMaxH3Model

        model = MiniMaxH3Model(
            model_filename=model_filename,
            model_def=model_def or {},
            text_encoder_filename=text_encoder_filename,
            dtype=dtype,
        )
        pipe = {
            "transformer": model.transformer,
            # Keep the wrapper top-level so MMGP's forward hook moves both
            # the truncated language model and vision tower together.
            "text_encoder": model.conditioner,
            "vae": model.vae,
            "audio_vae": model.audio_vae,
        }
        return model, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(
            {
                "num_inference_steps": 20,
                "video_length": 124,
                "resolution": "864x480",
                "guidance_scale": 1.0,
                "image_prompt_type": "",
            }
        )

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        # Saved settings created before this family existed cannot need a
        # migration, but imported presets still need valid H3 geometry.
        from .packing import align_num_frames

        try:
            requested_frames = int(ui_defaults.get("video_length", 124))
        except (TypeError, ValueError):
            requested_frames = 124
        aligned_frames = align_num_frames(max(1, requested_frames))
        ui_defaults["video_length"] = min(345, max(124, aligned_frames))
        resolution = str(ui_defaults.get("resolution", "864x480"))
        if resolution not in {value for _, value in _RESOLUTIONS}:
            ui_defaults["resolution"] = "864x480"
        ui_defaults["guidance_scale"] = 1.0
