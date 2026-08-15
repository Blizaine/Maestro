from __future__ import annotations

import os
from pathlib import Path

import torch

from shared.utils import files_locator as fl


UPSTREAM_REPO = "Lightricks/LTX-2.5"
UPSTREAM_REVISION = "28dac7acdc1f78a70e98687db261a949754f8941"

TRANSFORMER_FILENAME = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
TEXT_ENCODER_FILENAME = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
FAST_VIDEO_VAE_FILENAME = "ltx-2.5-video-vae-conv-bf16.safetensors"
NAD_VIDEO_VAE_FILENAME = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE_FILENAME = "ltx-2.5-audio-vae-bf16.safetensors"
SPATIAL_UPSCALER_FILENAME = (
    "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
)

# Native WanGP/MMGP assets.  Unlike the isolated official runtime, this stack
# remains profiled in the Maestro process between jobs and uses MMGP's dynamic
# LoRA injection—the same path validated by current Wan2GP.
NATIVE_REPO = "DeepBeepMeep/LTX-2"
NATIVE_BASE_MODEL_TYPE = "ltx2_25_22B"
NATIVE_CONFIG_FILENAME = "ltx2_25_22b_config.json"
NATIVE_SPATIAL_UPSCALER_FILENAME = (
    "ltx-2.5-spatial-upscaler-x2-1.0_bf16.safetensors"
)
NATIVE_FAST_VIDEO_VAE_FILENAME = "ltx-2.5-22b_video_vae_bf16.safetensors"
NATIVE_NAD_VIDEO_VAE_FILENAME = (
    "ltx-2.5-22b_diffusion_video_vae_bf16.safetensors"
)
NATIVE_AUDIO_VAE_FILENAME = "ltx-2.5-22b_audio_vae_bf16.safetensors"
NATIVE_VOCODER_FILENAME = "ltx-2.5-22b_vocoder_bf16.safetensors"
NATIVE_TEXT_PROJECTION_FILENAME = (
    "ltx-2.5-22b_text_embedding_projection_bf16.safetensors"
)
NATIVE_VIDEO_CONNECTORS = {
    "bf16": "ltx-2.5-22b_video_embeddings_connector_bf16.safetensors",
    "int8": "ltx-2.5-22b_video_embeddings_connector_int8_convrot.safetensors",
    "nvfp4": "ltx-2.5-22b_video_embeddings_connector_nvfp4_bf16.safetensors",
}
NATIVE_AUDIO_CONNECTORS = {
    "bf16": "ltx-2.5-22b_audio_embeddings_connector_bf16.safetensors",
    "int8": "ltx-2.5-22b_audio_embeddings_connector_int8_convrot.safetensors",
    "nvfp4": "ltx-2.5-22b_audio_embeddings_connector_nvfp4_bf16.safetensors",
}
NATIVE_GEMMA_FOLDER = "gemma4-12b-ltx-v1"
NATIVE_GEMMA_BF16_FILENAME = f"{NATIVE_GEMMA_FOLDER}_bf16.safetensors"
NATIVE_GEMMA_INT8_FILENAME = f"{NATIVE_GEMMA_FOLDER}_int8_convrot.safetensors"
NATIVE_GEMMA_TOKENIZER_FILES = [
    "config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
]

VIDEO_VAE_VARIANTS = {
    "fast": {
        "filename": NATIVE_FAST_VIDEO_VAE_FILENAME,
        "label": "Fast VAE (Recommended)",
        "description": (
            "LTX-2.5's original convolutional decoder. It is substantially "
            "faster and uses less VRAM than the optional diffusion decoder."
        ),
        "experimental": False,
    },
    "nad": {
        "filename": NATIVE_NAD_VIDEO_VAE_FILENAME,
        "label": "NAD Diffusion VAE",
        "description": (
            "Optional higher-quality diffusion decoder. It is much slower "
            "and can require substantially more VRAM."
        ),
        "experimental": True,
    },
}


# The official two-stage pipeline generates at half resolution, then performs
# latent x2 spatial upscaling. Both target dimensions must therefore be
# divisible by 64 (rather than the single-stage graph's divisor of 32).
_LTX25_RESOLUTION_PRESETS = {
    "480p": {
        "label": "480p",
        "values": {
            "auto": "auto_480p",
            "16:9": "896x512",
            "9:16": "512x896",
            "1:1": "512x512",
            "4:3": "704x512",
            "3:4": "512x704",
        },
    },
    "540p": {
        "label": "540p",
        "hint": "Uses the exact 16:9, two-stage-aligned 1024x576 canvas.",
        "values": {
            "auto": "auto_540p",
            "16:9": "1024x576",
            "9:16": "576x1024",
            "1:1": "576x576",
            "4:3": "768x576",
            "3:4": "576x768",
        },
    },
    "720p": {
        "label": "720p",
        "hint": "Uses a two-stage-aligned 1280x704 canvas.",
        "values": {
            "auto": "auto_720p",
            "16:9": "1280x704",
            "9:16": "704x1280",
            "1:1": "704x704",
            "4:3": "960x704",
            "3:4": "704x960",
        },
    },
    "1080p": {
        "label": "1080p",
        "experimental": True,
        "hint": (
            "High-resolution two-stage generation. Start with 10 seconds "
            "or less on 24 GB GPUs."
        ),
        "values": {
            "auto": "auto_1080p",
            "16:9": "1920x1088",
            "9:16": "1088x1920",
            "1:1": "1088x1088",
            "4:3": "1472x1088",
            "3:4": "1088x1472",
        },
    },
}
_LTX25_RESOLUTION_PRESET_ORDER = ["480p", "540p", "720p", "1080p"]
_LTX25_AUTO_RESOLUTION_BUDGETS = {
    "auto": 1280 * 704,
    "auto_480p": 896 * 512,
    "auto_540p": 1024 * 576,
    "auto_720p": 1280 * 704,
    "auto_1080p": 1920 * 1088,
}
_LTX25_AUTO_RESOLUTION_FALLBACKS = {
    "auto": "1280x704",
    "auto_480p": "896x512",
    "auto_540p": "1024x576",
    "auto_720p": "1280x704",
    "auto_1080p": "1920x1088",
}


def normalize_video_vae_variant(value: object) -> str:
    normalized = str(value or "fast").strip().lower()
    aliases = {
        "": "fast",
        "conv": "fast",
        "convolutional": "fast",
        FAST_VIDEO_VAE_FILENAME.lower(): "fast",
        NATIVE_FAST_VIDEO_VAE_FILENAME.lower(): "fast",
        "diffusion": "nad",
        NAD_VIDEO_VAE_FILENAME.lower(): "nad",
        NATIVE_NAD_VIDEO_VAE_FILENAME.lower(): "nad",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in VIDEO_VAE_VARIANTS:
        raise ValueError(
            f"Unknown LTX-2.5 video VAE '{value}'. Choose 'fast' or 'nad'."
        )
    return normalized


def _native_connector_variant(transformer_path: str | None) -> str:
    name = os.path.basename(str(transformer_path or "")).lower()
    if "nvfp4" in name:
        return "nvfp4"
    if "int8" in name or "quanto" in name:
        return "int8"
    return "bf16"


def _native_component_paths(
    transformer_path: str,
    video_vae_variant: str,
) -> dict[str, str]:
    connector_variant = _native_connector_variant(transformer_path)
    config_path = (
        Path(__file__).resolve().parents[1]
        / "ltx2"
        / "configs"
        / NATIVE_CONFIG_FILENAME
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing native LTX-2.5 config: {config_path}")
    return {
        "transformer": transformer_path,
        "video_vae": fl.locate_file(
            str(VIDEO_VAE_VARIANTS[video_vae_variant]["filename"])
        ),
        "audio_vae": fl.locate_file(NATIVE_AUDIO_VAE_FILENAME),
        "vocoder": fl.locate_file(NATIVE_VOCODER_FILENAME),
        "text_embedding_projection": fl.locate_file(
            NATIVE_TEXT_PROJECTION_FILENAME
        ),
        "video_embeddings_connector": fl.locate_file(
            NATIVE_VIDEO_CONNECTORS[connector_variant]
        ),
        "audio_embeddings_connector": fl.locate_file(
            NATIVE_AUDIO_CONNECTORS[connector_variant]
        ),
        "spatial_upsampler": fl.locate_file(
            NATIVE_SPATIAL_UPSCALER_FILENAME
        ),
        "model_config": str(config_path),
    }


def _locate_component(folder: str, filename: str) -> str:
    candidates = (
        os.path.join("ltx25", folder, filename),
        os.path.join(folder, filename),
        filename,
    )
    for candidate in candidates:
        located = fl.locate_file(candidate, error_if_none=False)
        if located:
            return located
    raise FileNotFoundError(
        f"LTX-2.5 component '{filename}' is missing. Maestro searched the "
        f"managed checkpoint folders for: {', '.join(candidates)}. Maestro's "
        "managed LTX-2.5 files are public and require no account. Check the "
        "earlier download error, internet connection, and free disk space, "
        "then retry the model download or run Update."
    )


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["ltx2_25"]

    @staticmethod
    def query_family_maps():
        # Keep checkpoint components separate. LoRA compatibility is exposed
        # explicitly through get_lora_dir() below; treating the two full model
        # architectures as interchangeable would make checkpoint resolution
        # unsafe.
        return {}, {}

    @staticmethod
    def query_model_family():
        return "ltx25"

    @staticmethod
    def query_family_infos():
        return {"ltx25": (39, "LTX-2.5")}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        pipeline_kind = str(
            model_def.get("ltx2_pipeline") or "distilled"
        ).strip().lower()
        distilled = pipeline_kind == "distilled"
        model_features = {
            "text_encoder_folder": NATIVE_GEMMA_FOLDER,
            "text_encoder_URLs": [
                (
                    f"https://huggingface.co/{NATIVE_REPO}/resolve/main/"
                    f"{NATIVE_GEMMA_FOLDER}/{NATIVE_GEMMA_BF16_FILENAME}"
                ),
                (
                    f"https://huggingface.co/{NATIVE_REPO}/resolve/main/"
                    f"{NATIVE_GEMMA_FOLDER}/{NATIVE_GEMMA_INT8_FILENAME}"
                ),
            ],
            "dtype": "bf16",
            "fps": 24,
            "frames_minimum": 17,
            "frames_steps": 8,
            "block_size": 64,
            "sliding_window": True,
            "multi_window_sequence_controls": True,
            "video_continuation": True,
            "image_prompt_types_allowed": "TSEVL",
            "end_frames_always_enabled": True,
            "sliding_window_end_image_at_final": True,
            "returns_audio": True,
            "auto_null_audio": True,
            "multimedia_generation": True,
            "any_audio_prompt": True,
            "audio_prompt_choices": True,
            "one_speaker_only": True,
            "audio_guide_label": "Audio Prompt (Soundtrack)",
            "audio_scale_name": "Prompt Audio Strength",
            "audio_prompt_type_sources": {
                "selection": ["", "A", "K", "2"],
                "labels": {
                    "": "Generate Video & Soundtrack based on Text Prompt",
                    "A": "Generate Video based on Soundtrack and Text Prompt",
                    "K": (
                        "Generate Video based on Control Video + its Audio "
                        "Track and Text Prompt"
                    ),
                    "2": "Generate Audio based on Control Video and Text Prompt",
                },
                "custom_flags": {
                    "2": "Generate Audio based on Control Video and Text Prompt",
                },
                "letters_filter": "AK2",
                "show_label": False,
                "default": "",
            },
            "audio_guide_window_slicing": True,
            # A standalone soundtrack tile is unambiguous for this family:
            # it means Audio-to-Video. Older saved settings (and a model-switch
            # race in the React UI) can retain the uploaded path while losing
            # the hidden ``A`` selector. Let the endpoint/WanGP repair that
            # inconsistent state when no Control Video is attached.
            "infer_audio_prompt_from_guide": True,
            "video_length_not_limited_by_audio": True,
            "output_audio_is_input_audio": True,
            "sliding_window_audio_history": True,
            "extra_control_frames": 1,
            "dont_cat_preguide": True,
            "guide_custom_choices": {
                "choices": [
                    ("No Control Video", ""),
                    ("Raw Control Video", "VG"),
                    ("Inject Frames", "KFI"),
                ],
                "letters_filter": "VGKFI",
                "default": "",
                "label": "Control Video / Frames Injection",
            },
            "custom_frames_injection": True,
            "one_image_ref_only": True,
            "mask_preprocessing": {"selection": [""], "visible": False},
            "custom_denoising_strength": True,
            "denoising_strength": {
                "label": (
                    "Control Video Strength (higher = closer to the Control "
                    "Video)"
                ),
                "name": "Control Video Strength",
            },
            "sliding_window_defaults": {
                "overlap_min": 1,
                "overlap_max": 97,
                "overlap_step": 8,
                "overlap_default": 9,
                "window_min": 17,
                "window_max": 501,
                "window_step": 8,
                "window_default": 241,
                "discard_last_frames": 8,
            },
            "director_audio_input_mode": "generic_audio_guide",
            "director_reference_mode": "start_frame",
            "director_voice_reference_mode": "none",
            "guidance_max_phases": 2,
            "visible_phases": 0 if distilled else 1,
            "lock_guidance_phases": True,
            "no_negative_prompt": distilled,
            "NAG": True,
            "compile": False,
            "input_video_strength": "Image conditioning strength",
            "ltx2_pipeline": pipeline_kind,
            "ltx2_22B_class": True,
            "ltx2_spatial_upscaler_file": NATIVE_SPATIAL_UPSCALER_FILENAME,
            "ltx25_native_runtime": True,
            "ltx25_video_vae_choices": [
                {"value": value, **spec}
                for value, spec in VIDEO_VAE_VARIANTS.items()
            ],
            "ltx25_video_vae_default": "fast",
            "resolution_presets": _LTX25_RESOLUTION_PRESETS,
            "resolution_preset_order": _LTX25_RESOLUTION_PRESET_ORDER,
            "supports_auto_aspect": True,
            "auto_resolution_budgets": _LTX25_AUTO_RESOLUTION_BUDGETS,
            "auto_resolution_fallbacks": _LTX25_AUTO_RESOLUTION_FALLBACKS,
            "profiles_dir": ["ltx2_25"],
        }
        if distilled:
            model_features.update(
                {
                    "lock_inference_steps": True,
                    "lock_guidance_scale": True,
                    "lora_compatibility_note": (
                        "LTX-2.5 shares Maestro's LTX-2/2.3 LoRA folder. "
                        "Standard LTX-2/2.3 adapters can be selected directly; "
                        "specialized IC-LoRAs still need their matching control "
                        "workflow. LTX-2.3 distilled accelerator LoRAs are "
                        "unnecessary because this checkpoint is already "
                        "distilled."
                    ),
                }
            )
        else:
            model_features.update(
                {
                    "audio_guidance": True,
                    "adaptive_projected_guidance": True,
                    "cfg_star": True,
                    "perturbation": True,
                    "alt_guidance": "Modality Guidance",
                    "alt_scale": "Guidance Rescale",
                    "perturbation_choices": [
                        ("Off", 0),
                        ("Skip Layer Guidance", 1),
                        ("Skip Self Attention", 2),
                    ],
                    "perturbation_layers_max": 48,
                    "lora_compatibility_note": (
                        "LTX-2.5 shares Maestro's LTX-2/2.3 LoRA folder. "
                        "Standard LTX-2/2.3 adapters can be selected directly; "
                        "specialized IC-LoRAs still need their matching control "
                        "workflow. Distilled accelerator LoRAs are not intended "
                        "for the Dev checkpoint unless their author explicitly "
                        "marks them compatible."
                    ),
                }
            )
        return model_features

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        # LTX-2.5 intentionally shares --lora-dir-ltx2 with LTX-2/2.3.
        # The LTX-2 handler owns that CLI option, so registering it again here
        # would make argparse reject startup with a duplicate option.
        del parser, lora_root

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        return getattr(args, "lora_dir_ltx2", None) or os.path.join(
            lora_root, "ltx2"
        )

    @staticmethod
    def get_vae_block_size(base_model_type):
        return 64

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        model_def = model_def or {}
        model_urls = model_def.get("URLs", [])
        if isinstance(model_urls, str):
            model_urls = [model_urls]
        variants = {
            _native_connector_variant(url)
            for url in model_urls
        } or {"int8"}
        component_files = [
            NATIVE_SPATIAL_UPSCALER_FILENAME,
            NATIVE_FAST_VIDEO_VAE_FILENAME,
            NATIVE_NAD_VIDEO_VAE_FILENAME,
            NATIVE_AUDIO_VAE_FILENAME,
            NATIVE_VOCODER_FILENAME,
            NATIVE_TEXT_PROJECTION_FILENAME,
        ]
        for variant in sorted(variants):
            component_files.extend(
                [
                    NATIVE_VIDEO_CONNECTORS[variant],
                    NATIVE_AUDIO_CONNECTORS[variant],
                ]
            )
        return [
            {
                "repoId": NATIVE_REPO,
                "sourceFolderList": [""],
                "fileList": [component_files],
            },
            {
                "repoId": NATIVE_REPO,
                "sourceFolderList": [NATIVE_GEMMA_FOLDER],
                "fileList": [NATIVE_GEMMA_TOKENIZER_FILES],
            },
        ]

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        frame_count = int(inputs.get("video_length") or 0)
        if frame_count > 0 and (frame_count - 1) % 8:
            # Saved settings and media-derived durations may arrive as a raw
            # frame count. Repair them here rather than rejecting an old run.
            inputs["video_length"] = max(
                9,
                int(round((frame_count - 1) / 8.0)) * 8 + 1,
            )
        pipeline_kind = str(
            model_def.get("ltx2_pipeline") or "distilled"
        ).strip().lower()
        if pipeline_kind == "distilled":
            inputs.update(
                {
                    "num_inference_steps": 8,
                    "guidance_scale": 1.0,
                    "audio_guidance_scale": 1.0,
                    "alt_guidance_scale": 1.0,
                    "alt_scale": 0.0,
                }
            )
        else:
            # The LTX-2.5 Dev pipeline is currently validated with Euler.
            # Repair stale settings inherited from another model instead of
            # failing after the large checkpoint has loaded.
            inputs["sample_solver"] = "euler"
            inputs["self_refiner_setting"] = 0

    @staticmethod
    def load_model(
        model_filename,
        model_type,
        base_model_type,
        model_def,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        from models.ltx2.ltx2 import LTX2

        if isinstance(model_filename, (list, tuple)):
            transformer_path = next((str(item) for item in model_filename if item), "")
        else:
            transformer_path = str(model_filename or "")
        if not transformer_path or not Path(transformer_path).is_file():
            raise FileNotFoundError(
                "The native LTX-2.5 transformer checkpoint is missing. "
                "Run Update/Install Models and retry."
            )

        video_vae_variant = normalize_video_vae_variant(
            kwargs.get("ltx25_video_vae")
            or model_def.get("ltx25_video_vae_default", "fast")
        )
        video_vae_spec = VIDEO_VAE_VARIANTS[video_vae_variant]
        print(
            "[LTX-2.5] Video decoder: "
            f"{video_vae_spec['label']}."
        )

        native_model_def = {
            **model_def,
            "ltx2_pipeline": str(
                model_def.get("ltx2_pipeline") or "distilled"
            ).strip().lower(),
            "ltx2_spatial_upscaler_file": NATIVE_SPATIAL_UPSCALER_FILENAME,
        }
        component_paths = _native_component_paths(
            transformer_path,
            video_vae_variant,
        )
        model = LTX2(
            model_filename=transformer_path,
            model_type=model_type,
            base_model_type=NATIVE_BASE_MODEL_TYPE,
            model_def=native_model_def,
            dtype=dtype,
            VAE_dtype=kwargs.get("VAE_dtype", torch.float32),
            text_encoder_filename=kwargs.get("text_encoder_filename"),
            text_encoder_filepath=NATIVE_GEMMA_FOLDER,
            checkpoint_paths=component_paths,
        )
        model.video_vae_variant = video_vae_variant

        pipe = {
            "transformer": model.model,
            "text_encoder": model.text_encoder,
            "text_embedding_projection": model.text_embedding_projection,
            "vae": model.video_decoder,
            "video_encoder": model.video_encoder,
            "audio_encoder": model.audio_encoder,
            "audio_decoder": model.audio_decoder,
            "vocoder": model.vocoder,
            "spatial_upsampler": model.spatial_upsampler,
        }
        if model.split_text_connectors:
            pipe["video_embeddings_connector"] = (
                model.text_embeddings_connector.video_embeddings_connector
            )
            pipe["audio_embeddings_connector"] = (
                model.text_embeddings_connector.audio_embeddings_connector
            )
        else:
            pipe["text_embeddings_connector"] = model.text_embeddings_connector
        return model, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        pipeline_kind = str(
            model_def.get("ltx2_pipeline") or "distilled"
        ).strip().lower()
        common_defaults = {
            "video_length": 241,
            "resolution": "1024x576",
            "input_video_strength": 0.7,
            "image_prompt_type": "",
            "audio_prompt_type": "",
            "sliding_window_size": 241,
            "sliding_window_overlap": 9,
            "sliding_window_discard_last_frames": 8,
            "denoising_strength": 1.0,
        }
        if pipeline_kind == "distilled":
            common_defaults.update(
                {
                    "num_inference_steps": 8,
                    "guidance_scale": 1.0,
                    "audio_guidance_scale": 1.0,
                    "alt_guidance_scale": 1.0,
                    "alt_scale": 0.0,
                    "guidance_phases": 1,
                }
            )
        else:
            common_defaults.update(
                {
                    "num_inference_steps": 30,
                    "guidance_scale": 3.0,
                    "audio_guidance_scale": 7.0,
                    "alt_guidance_scale": 3.0,
                    "alt_scale": 0.7,
                    "sample_solver": "euler",
                    "perturbation_layers": [28],
                    "perturbation_start_perc": 0,
                    "perturbation_end_perc": 100,
                    "apg_switch": 0,
                    "cfg_star_switch": 0,
                    "guidance_phases": 2,
                    "self_refiner_setting": 0,
                }
            )
        ui_defaults.update(common_defaults)

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        pipeline_kind = str(
            model_def.get("ltx2_pipeline") or "distilled"
        ).strip().lower()
        if pipeline_kind == "distilled":
            ui_defaults["num_inference_steps"] = 8
            ui_defaults["guidance_scale"] = 1.0
            ui_defaults["audio_guidance_scale"] = 1.0
            ui_defaults["alt_guidance_scale"] = 1.0
            ui_defaults["alt_scale"] = 0.0
            ui_defaults.setdefault("guidance_phases", 1)
        else:
            ui_defaults.setdefault("num_inference_steps", 30)
            ui_defaults.setdefault("guidance_scale", 3.0)
            ui_defaults.setdefault("audio_guidance_scale", 7.0)
            ui_defaults.setdefault("alt_guidance_scale", 3.0)
            ui_defaults.setdefault("alt_scale", 0.7)
            ui_defaults["sample_solver"] = "euler"
            ui_defaults.setdefault("perturbation_layers", [28])
            ui_defaults.setdefault("perturbation_start_perc", 0)
            ui_defaults.setdefault("perturbation_end_perc", 100)
            ui_defaults.setdefault("apg_switch", 0)
            ui_defaults.setdefault("cfg_star_switch", 0)
            ui_defaults["guidance_phases"] = 2
            ui_defaults["self_refiner_setting"] = 0
        ui_defaults.setdefault("input_video_strength", 0.7)
        ui_defaults.setdefault("audio_prompt_type", "")
        ui_defaults.setdefault("sliding_window_size", 241)
        ui_defaults.setdefault("sliding_window_overlap", 9)
        ui_defaults.setdefault("sliding_window_discard_last_frames", 8)
        ui_defaults.setdefault("denoising_strength", 1.0)
