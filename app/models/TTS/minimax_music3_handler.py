"""Maestro family handler for MiniMax-Music3."""

from __future__ import annotations

import os

import torch

from shared.utils import files_locator as fl


MODEL_TYPE = "minimax_music3"
REPO_ID = "MiniMaxAI/MiniMax-Music3"
REVISION = "bd348f9c49ea3c1b39f33ace3436f8fad435f24e"
ASSET_ROOT = "minimax_music3"
DEFAULT_DURATION_SECONDS = 120


def _download_definition():
    return {
        "repoId": REPO_ID,
        "revision": REVISION,
        "sourceFolderList": [
            "",
            "tokenizer",
            "language_model",
            "rvq_depth_decoder",
            "condition_encoder",
            "transformer",
            "scheduler",
            "vocoder",
        ],
        "targetFolderList": [ASSET_ROOT] * 8,
        "fileList": [
            ["modular_model_index.json", "LICENSE"],
            ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json"],
            [
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "model-00001-of-00004.safetensors",
                "model-00002-of-00004.safetensors",
                "model-00003-of-00004.safetensors",
                "model-00004-of-00004.safetensors",
            ],
            ["config.json", "diffusion_pytorch_model.safetensors"],
            ["config.json", "diffusion_pytorch_model.safetensors"],
            [
                "config.json",
                "diffusion_pytorch_model.safetensors.index.json",
                "diffusion_pytorch_model-00001-of-00002.safetensors",
                "diffusion_pytorch_model-00002-of-00002.safetensors",
            ],
            ["scheduler_config.json"],
            ["config.json", "diffusion_pytorch_model.safetensors"],
        ],
    }


def _model_definition():
    return {
        "audio_only": True,
        "image_outputs": False,
        "sliding_window": False,
        "guidance_max_phases": 0,
        "lock_guidance_scale": True,
        "no_negative_prompt": True,
        "inference_steps": True,
        "temperature": False,
        "image_prompt_types_allowed": "",
        "supports_early_stop": True,
        "profiles_dir": [ASSET_ROOT],
        "compile": False,
        "dtype": "bf16",
        "prompt_class": "Lyrics",
        "prompt_description": (
            "Lyrics with section tags such as [Verse], [Chorus], [Bridge], "
            "[Instrumental], and [Outro]."
        ),
        "alt_prompt": {
            "label": "Structured Music Caption",
            "name": "Music Caption",
            "placeholder": (
                "### Global Metadata\nGenre, BPM, key, mood, and production...\n\n"
                "### Vocal Details\nVoice, delivery, harmonies, and effects...\n\n"
                "### Arrangement\nInstruments and section-by-section evolution..."
            ),
            "lines": 10,
        },
        "duration_slider": {
            "label": "Song duration (seconds)",
            "min": 5,
            "max": 300,
            "increment": 1,
            "default": DEFAULT_DURATION_SECONDS,
        },
        "music3_structured_caption": True,
        "music_caption_label": "Structured Music Caption",
        "music_caption_help": (
            "MiniMax-Music3 follows Global Metadata, Vocal Details, and "
            "Arrangement sections for detailed long-form control."
        ),
        "music_lyrics_help": (
            "Put section tags on their own lines. Supported tags include "
            "Intro, Verse, Pre-Chorus, Chorus, Post-Chorus, Bridge, "
            "Instrumental, Solo, and Outro."
        ),
    }


class family_handler:
    @staticmethod
    def query_supported_types():
        return [MODEL_TYPE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "tts"

    @staticmethod
    def query_family_infos():
        return {"tts": (200, "TTS")}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-minimax-music3",
            type=str,
            default=None,
            help=(
                "Reserved MiniMax-Music3 LoRA directory "
                f"(default: {os.path.join(lora_root, 'minimax_music3_music')})"
            ),
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_minimax_music3", None) or os.path.join(
            lora_root, "minimax_music3_music"
        )

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return _model_definition()

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return _download_definition()

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
        dtype=None,
        profile=0,
        **kwargs,
    ):
        from .minimax_music3 import MiniMaxMusic3Pipeline

        asset_root = fl.locate_folder(ASSET_ROOT, error_if_none=False)
        if asset_root is None:
            asset_root = os.path.join(fl.get_download_location(), ASSET_ROOT)
        pipeline = MiniMaxMusic3Pipeline.from_pretrained(
            asset_root,
            dtype=dtype or torch.bfloat16,
        )
        pipeline._maestro_mmgp_profile = profile
        pipe = {
            "language_model": pipeline.language_model,
            "rvq_depth_decoder": pipeline.rvq_depth_decoder,
            "condition_encoder": pipeline.condition_encoder,
            "transformer": pipeline.transformer,
            "vocoder": pipeline.vocoder,
        }
        return pipeline, {
            "pipe": pipe,
            "coTenantsMap": {
                "language_model": ["rvq_depth_decoder"],
                "rvq_depth_decoder": ["language_model"],
            },
            "workingVRAM": {
                "language_model": 4096,
                "rvq_depth_decoder": 1024,
                "transformer": 4096,
                "vocoder": 1024,
            },
        }

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        duration = model_def.get("duration_slider", {}).get(
            "default", DEFAULT_DURATION_SECONDS
        )
        ui_defaults.update(
            {
                "prompt": (
                    "[Verse]\nMorning light filters through the pines\n"
                    "Every quiet road is yours and mine\n"
                    "[Chorus]\nSoftly the whole world starts to breathe\n"
                    "Stay for one more song with me\n[Outro]"
                ),
                "alt_prompt": (
                    "### Global Metadata\nWarm acoustic pop at 96 BPM in C major; "
                    "intimate and hopeful, growing into a wide final chorus; "
                    "polished natural production.\n\n"
                    "### Vocal Details\nSoft, close female lead with breathy verses, "
                    "clear diction, and light stacked harmonies in the chorus.\n\n"
                    "### Arrangement\nFingerpicked acoustic guitar and soft piano open "
                    "the song. Brushed drums and upright bass enter in the chorus; "
                    "strings bloom gently before a sparse outro."
                ),
                "audio_prompt_type": "",
                "duration_seconds": duration,
                "video_length": 0,
                "num_inference_steps": 30,
                "guidance_scale": 1.7,
                "negative_prompt": "",
                "repeat_generation": 1,
                "multi_prompts_gen_type": 2,
            }
        )

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        ui_defaults.setdefault("audio_prompt_type", "")
        ui_defaults.setdefault("num_inference_steps", 30)
        ui_defaults.setdefault(
            "duration_seconds",
            model_def.get("duration_slider", {}).get(
                "default", DEFAULT_DURATION_SECONDS
            ),
        )
        ui_defaults.setdefault("guidance_scale", 1.7)
        ui_defaults.setdefault("alt_prompt", "")

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        lyrics = str(one_prompt or "").strip()
        caption = str(inputs.get("alt_prompt") or "").strip()
        if not lyrics:
            return (
                "MiniMax-Music3 requires lyrics. Use [Instrumental] for an "
                "instrumental song."
            )
        if not caption:
            return "MiniMax-Music3 requires a Music Caption."
        if inputs.get("audio_guide") is not None or inputs.get("audio_guide2") is not None:
            return "MiniMax-Music3 does not support reference audio."
        return None

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        try:
            duration = float(
                inputs.get("duration_seconds", DEFAULT_DURATION_SECONDS)
            )
        except (TypeError, ValueError):
            return "MiniMax-Music3 duration must be a number between 5 and 300 seconds."
        if duration < 5 or duration > 300:
            return "MiniMax-Music3 duration must be between 5 and 300 seconds."
        try:
            steps = int(inputs.get("num_inference_steps", 30))
        except (TypeError, ValueError):
            return "MiniMax-Music3 inference steps must be an integer."
        if steps < 1 or steps > 100:
            return "MiniMax-Music3 inference steps must be between 1 and 100."
        return None
