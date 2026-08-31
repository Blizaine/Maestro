"""Maestro family handler for MiniMax-Music3."""

from __future__ import annotations

import os

import torch

from shared.utils import files_locator as fl
from .minimax_music3.prompting import validate_music3_lyrics


MODEL_TYPE = "minimax_music3"
OFFICIAL_REPO_ID = "MiniMaxAI/MiniMax-Music3"
OFFICIAL_REVISION = "bd348f9c49ea3c1b39f33ace3436f8fad435f24e"
OPTIMIZED_REPO_ID = "DeepBeepMeep/TTS"
OPTIMIZED_REVISION = "d31b4665414200fcab779ced520b01bd9f5e07ba"
PROJECT_FOLDER = "MiniMax-Music3"
TEXT_ENCODER_FOLDER = "MiniMaxMusic3-Qwen3"
TEXT_ENCODER_BF16 = "MiniMaxMusic3-Qwen3_bf16.safetensors"
TEXT_ENCODER_INT8 = "MiniMaxMusic3-Qwen3_int8_convrot.safetensors"
DEFAULT_DURATION_SECONDS = 120

PROJECT_FILES = [
    "rvq_depth_decoder_config.json",
    "rvq_depth_decoder_bf16.safetensors",
    "rvq_depth_decoder_int8_convrot.safetensors",
    "condition_encoder_config.json",
    "condition_encoder_fp32.safetensors",
    "vocoder_config.json",
    "vocoder_fp32.safetensors",
    "scheduler_config.json",
]
TOKENIZER_FILES = [
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
]


def _asset_path(filename: str) -> str:
    return fl.locate_file(os.path.join(PROJECT_FOLDER, filename))


def _source_config(filename: str) -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "minimax_music3",
        filename,
    )


def _optimized_url(*parts: str) -> str:
    path = "/".join(str(part).strip("/") for part in parts)
    return (
        f"https://huggingface.co/{OPTIMIZED_REPO_ID}/resolve/"
        f"{OPTIMIZED_REVISION}/{path}"
    )


def _model_definition():
    return {
        "group": "music",
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
        "profiles_dir": [MODEL_TYPE],
        "text_encoder_URLs": [
            _optimized_url(
                TEXT_ENCODER_FOLDER,
                TEXT_ENCODER_BF16,
            ),
            _optimized_url(
                TEXT_ENCODER_FOLDER,
                TEXT_ENCODER_INT8,
            ),
        ],
        "text_encoder_folder": TEXT_ENCODER_FOLDER,
        "lm_engines": ["cg", "vllm"],
        "music3_accelerated_semantics": True,
        "compile": False,
        "dtype": "bf16",
        "prompt_class": "Lyrics",
        "prompt_description": (
            "Lyrics with bare section tags such as [Verse], [Chorus], "
            "[Bridge], [Guitar Solo], [Instrumental], and [Outro]. Put all "
            "production directions in the Music Caption."
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
            "Use bare section tags on their own lines. Never put mood, "
            "instruments, timing, or stage directions inside brackets; Music3 "
            "may sing them aloud. Use [Instrumental] or [Guitar Solo] with no "
            "lyric lines beneath it for an instrumental passage."
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
        del computeList, base_model_type, model_def
        return [
            {
                "repoId": OPTIMIZED_REPO_ID,
                "revision": OPTIMIZED_REVISION,
                "sourceFolderList": [PROJECT_FOLDER, TEXT_ENCODER_FOLDER],
                "fileList": [PROJECT_FILES, TOKENIZER_FILES],
            },
            {
                "repoId": OFFICIAL_REPO_ID,
                "revision": OFFICIAL_REVISION,
                "sourceFolderList": [""],
                "targetFolderList": [PROJECT_FOLDER],
                "fileList": [["LICENSE"]],
            },
        ]

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=None,
        VAE_dtype=None,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        profile=0,
        lm_decoder_engine="legacy",
        **kwargs,
    ):
        del (
            base_model_type,
            model_def,
            quantizeTransformer,
            text_encoder_quantization,
            VAE_dtype,
            mixed_precision_transformer,
            kwargs,
        )
        from .minimax_music3.optimized_pipeline import MiniMaxMusic3Pipeline

        if isinstance(model_filename, (list, tuple)):
            if not model_filename:
                raise RuntimeError("MiniMax Music3 transformer checkpoint is missing.")
            flow_weights = model_filename[0]
        else:
            flow_weights = model_filename
        if not flow_weights:
            raise RuntimeError("MiniMax Music3 transformer checkpoint is missing.")
        if not text_encoder_filename:
            raise RuntimeError("MiniMax Music3 Qwen checkpoint is missing.")

        accelerated = str(lm_decoder_engine or "legacy").lower() in {
            "cg",
            "vllm",
        }
        rvq_weights = {
            "bf16": _asset_path("rvq_depth_decoder_bf16.safetensors"),
            "int8": _asset_path(
                "rvq_depth_decoder_int8_convrot.safetensors"
            ),
        }
        asset_paths = {
            "text_encoder_config": _source_config("text_encoder_config.json"),
            "transformer_config": _source_config("transformer_config.json"),
            "rvq_config": _asset_path("rvq_depth_decoder_config.json"),
            "rvq_weights": rvq_weights["int8" if accelerated else "bf16"],
            "condition_config": _asset_path("condition_encoder_config.json"),
            "condition_weights": _asset_path(
                "condition_encoder_fp32.safetensors"
            ),
            "vocoder_config": _asset_path("vocoder_config.json"),
            "vocoder_weights": _asset_path("vocoder_fp32.safetensors"),
            "scheduler_dir": os.path.dirname(
                _asset_path("scheduler_config.json")
            ),
            "tokenizer_dir": os.path.dirname(
                fl.locate_file(
                    os.path.join(TEXT_ENCODER_FOLDER, "tokenizer.json")
                )
            ),
        }
        # Validate the two remaining tokenizer assets now so a partial
        # download fails with the actual missing filename.
        fl.locate_file(
            os.path.join(TEXT_ENCODER_FOLDER, "tokenizer_config.json")
        )
        fl.locate_file(os.path.join(TEXT_ENCODER_FOLDER, "chat_template.jinja"))

        pipeline = MiniMaxMusic3Pipeline(
            flow_weights,
            text_encoder_filename,
            asset_paths,
            dtype or torch.bfloat16,
            lm_decoder_engine=lm_decoder_engine,
        )
        pipeline._maestro_mmgp_profile = profile
        pipeline._maestro_music3_engine = lm_decoder_engine
        if accelerated:
            # CUDA-graph semantic decoding requires Qwen and the RVQ decoder
            # to remain resident together. Their ConvRot checkpoints fit the
            # 16-20 GB tier where the former BF16 path streamed every frame.
            pipeline.text_encoder._budget = 0
            pipeline.rvq_depth_decoder._budget = 0

        if save_quantized:
            from wgp import save_quantized_model

            save_quantized_model(
                pipeline.transformer,
                model_type,
                flow_weights,
                dtype or torch.bfloat16,
                asset_paths["transformer_config"],
                submodel_no=1,
            )

        pipe = {
            "text_encoder": pipeline.text_encoder,
            "rvq_depth_decoder": pipeline.rvq_depth_decoder,
            "condition_encoder": pipeline.condition_encoder,
            "transformer": pipeline.transformer,
            "vocoder": pipeline.vocoder,
        }
        offload_def = {
            "pipe": pipe,
            "coTenantsMap": {
                "transformer": ["condition_encoder"],
                "condition_encoder": ["transformer"],
                "text_encoder": ["rvq_depth_decoder"],
                "rvq_depth_decoder": ["text_encoder"],
            },
        }
        if int(profile) in (2, 4, 5):
            offload_def["budgets"] = {
                "transformer": 400,
                "text_encoder": 800,
            }
        return pipeline, offload_def

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
        lyrics_error = validate_music3_lyrics(lyrics)
        if lyrics_error:
            return lyrics_error
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
