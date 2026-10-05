import os
import math
import torch
from shared.utils.hf import build_hf_url


LONGCAT_SLIDING_WINDOW_DEFAULTS = {
    # Use the existing 93-frame recipe for local inference. Longer timelines
    # continue through windows; they must not inherit another engine's 40s pass.
    "window_min": 17,
    "window_max": 93,
    "window_step": 4,
    "window_default": 93,
    "overlap_min": 1,
    "overlap_max": 13,
    "overlap_step": 4,
    "overlap_default": 13,
    "discard_last_frames": 0,
}


def normalize_longcat_window_params(params, model_def):
    """Bound each LongCat pass without shortening the requested timeline."""
    model_type = str(params.get("model_type") or "")
    architecture = str((model_def or {}).get("architecture") or model_type)
    if architecture not in {"longcat_video", "longcat_avatar"} and model_type not in {
        "longcat_video", "longcat_avatar", "longcat_avatar_multi",
    }:
        return False

    def frame_count(value, default, label, minimum):
        if value is None:
            return default
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"LongCat {label} must be a whole frame count") from None
        if isinstance(value, bool) or not math.isfinite(number) or not number.is_integer() or number < minimum:
            raise ValueError(f"LongCat {label} must be a whole frame count")
        return int(number)

    previous = {key: params.get(key) for key in (
        "sliding_window_size", "sliding_window_overlap", "sliding_window_discard_last_frames",
    )}
    frames = frame_count(params.get("sliding_window_size"), 93, "window size", 1)
    frames = max(17, min(93, 17 + ((frames - 17 + 2) // 4) * 4))
    overlap = frame_count(params.get("sliding_window_overlap"), 13, "overlap", 0)
    overlap = max(1, min(13, 1 + ((overlap - 1 + 2) // 4) * 4))
    params.update(sliding_window_size=frames, sliding_window_overlap=overlap,
                  sliding_window_discard_last_frames=0)
    return any(params[key] != value for key, value in previous.items())


def longcat_avatar_weight_budget(total_vram_gb, resolution, window_frames,
                                additional_reserve_gb=0, continuation=False):
    """Reserve activation workspace before placing Avatar weights on the GPU.

    Avatar modulation, Q/K normalization and rotary embeddings use FP32 even
    with INT8 weights. A 93-frame 720p pass spilled into WDDM shared memory
    with the full transformer resident on a 24 GB card. Allow 17 GB of
    workspace at that token count, scaling the variable portion by the
    VAE/patch token grid. Continuation adds a reference latent and separate
    reference/overlap/noise attention buffers; reserve another 3 GB at the
    same grid size. This changes weight residency, not model precision.
    """
    total_vram_gb = max(0.0, float(total_vram_gb or 0))
    width, height = 1280, 720
    try:
        requested_width, requested_height = map(int, str(resolution).lower().split("x"))
        if requested_width > 0 and requested_height > 0:
            width, height = requested_width, requested_height
    except (TypeError, ValueError):
        pass
    frames = max(1, int(window_frames or 93))
    latent_frames = 1 + (frames - 1) // 4
    reference_latent_frames = 1 if continuation else 0
    latent_frames += reference_latent_frames
    tokens = math.ceil(width / 16) * math.ceil(height / 16) * latent_frames
    continuation_reserve = 3.0 * tokens / 86400 if continuation else 0.0
    requested_reserve = (2.0 + 15.0 * tokens / 86400 + continuation_reserve
                         + max(0.0, additional_reserve_gb))
    # Keep a small streaming slice on cards where this resolution cannot fit.
    # The clamped estimate remains visible for diagnostics.
    reserve = min(requested_reserve, max(0.0, total_vram_gb - 3.5))
    return {
        "weight_budget_gb": max(0.0, total_vram_gb - reserve),
        "activation_reserve_gb": reserve,
        "requested_activation_reserve_gb": requested_reserve,
        "activation_reserve_clamped": reserve < requested_reserve,
        "window_tokens": tokens,
        "reference_latent_frames": reference_latent_frames,
        "continuation_reserve_gb": continuation_reserve,
    }


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["longcat_video", "longcat_avatar"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "longcat"

    @staticmethod
    def query_family_infos():
        return {"longcat": (60, "LongCat")}

    @staticmethod
    def register_lora_cli_args(parser, lora_root):
        parser.add_argument(
            "--lora-dir-longcat",
            type=str,
            default=None,
            help=f"Path to a directory that contains LongCat Video LoRAs (default: {os.path.join(lora_root, 'longcat')})",
        )
        parser.add_argument(
            "--lora-dir-longcat-avatar",
            type=str,
            default=None,
            help=f"Path to a directory that contains LongCat Avatar LoRAs (default: {os.path.join(lora_root, 'longcat_avatar')})",
        )

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        if base_model_type == "longcat_avatar":
            return getattr(args, "lora_dir_longcat_avatar", None) or os.path.join(lora_root, "longcat_avatar")
        return getattr(args, "lora_dir_longcat", None) or os.path.join(lora_root, "longcat")

    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {
            "frames_minimum": 5,
            "frames_steps": 4,
            "sliding_window": True,
            "sliding_window_defaults": dict(LONGCAT_SLIDING_WINDOW_DEFAULTS),
            "guidance_max_phases": 1,
            "image_prompt_types_allowed": "TSVL",
            "video_continuation": True,
            "sample_solvers": [
                ("Auto (Continuation = Enhanced HF)", "auto"),
                ("Default", ""),
                ("Enhanced HF", "enhance_hf"),
                ("Distill", "distill"),
            ],
        }
        text_encoder_folder = "umt5-xxl"
        extra_model_def["text_encoder_URLs"] = [
            build_hf_url("DeepBeepMeep/Wan2.1", text_encoder_folder, "models_t5_umt5-xxl-enc-bf16.safetensors"),
            build_hf_url("DeepBeepMeep/Wan2.1", text_encoder_folder, "models_t5_umt5-xxl-enc-quanto_int8.safetensors"),
        ]
        extra_model_def["text_encoder_folder"] = text_encoder_folder

        if base_model_type == "longcat_video":
            extra_model_def.update(
                {
                    "fps": 15,
                    "profiles_dir": ["longcat_video"],
                    "t2v_class": True,
                    "i2v_class": True,
                }
            )
        elif base_model_type == "longcat_avatar":
            extra_model_def.update(
                {
                    "fps": 16,
                    "profiles_dir": [base_model_type],
                    "audio_guidance": True,
                    "any_audio_prompt": True,
                    "audio_prompt_choices": True,
                    "audio_guide_label": "Voice to follow",
                    "audio_guide2_label": "Voice to follow #2",
                    "infer_audio_prompt_from_guide": True,
                    "max_image_refs": 1,
                    "t2v_class": True,
                    "i2v_class": True,
                    "image_ref_choices": {
                        "choices": [("None", ""), ("Anchor Reference Image", "KI")],
                        "letters_filter": "KI",
                        "visible": True,
                        "label": "Anchor Reference Image",
                    },
                    "reference_image_enabled": True,
                    "no_background_removal": True,
                    "image_prompt_types_allowed": "TSVL",
                }
            )
            # WanGP's classic UI creates its paired guide control dynamically
            # for multi-speaker checkpoints. A single-choice config is used
            # only for the one-guide Studio-compatible checkpoint.
            if not (model_def or {}).get("multi_speakers_only", False):
                extra_model_def["audio_prompt_type_sources"] = {
                    "selection": ["A"],
                    "labels": {"A": "Use voice to drive the avatar"},
                    "default": "A",
                    "letters_filter": "A",
                    "show_label": False,
                }


        return extra_model_def

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("wan")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        download_def = [
            {
                "repoId": "DeepBeepMeep/Wan2.1",
                "sourceFolderList": ["umt5-xxl", "chinese-wav2vec2-base"],
                "fileList": [
                    ["special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"],
                    [
                        "config.json",
                        "preprocessor_config.json",
                        "pytorch_model.bin",
                        "readme.txt",
                    ],
                ],
            }
        ]
        download_def += [
            {
                "repoId": "DeepBeepMeep/Wan2.1",
                "sourceFolderList": [""],
                "fileList": [["Wan2.1_VAE_bf16.safetensors"]],
            }
        ]
        return download_def

    @staticmethod
    def load_model(
        model_filename,
        model_type,
        base_model_type,
        model_def,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        **kwargs,
    ):
        from .longcat_main import LongCatModel

        longcat_model = LongCatModel(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type=model_type,
            model_def=model_def,
            base_model_type=base_model_type,
            text_encoder_filename=text_encoder_filename,
            quantizeTransformer=quantizeTransformer,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            mixed_precision_transformer=mixed_precision_transformer,
            save_quantized=save_quantized,
        )

        pipe = {
            "transformer": longcat_model.transformer,
            "vae": longcat_model.vae,
            "text_encoder": longcat_model.text_encoder.model,
        }
        if longcat_model.audio_encoder is not None:
            pipe["wav2vec"] = longcat_model.audio_encoder

        return longcat_model, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(
            {
                "guidance_scale": 4.0,
                "num_inference_steps": 50,
                "audio_guidance_scale": 4.0,
                "sliding_window_overlap": 13,
                "sliding_window_size": 93,                 
            }
        )
        if base_model_type == "longcat_video":
            ui_defaults.update({"video_length": 93})

        if base_model_type in ["longcat_avatar"]:
            ui_defaults.update({"video_length": 93, "video_prompt_type": ""})

        if ui_defaults.get("sample_solver", "") == "":
            ui_defaults["sample_solver"] = "auto"
