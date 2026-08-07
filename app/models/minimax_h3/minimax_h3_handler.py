"""Maestro family handler for MiniMax H3 Base FL2VA and Ref2VA."""

from __future__ import annotations

import os

import torch


_MODEL_TYPE = "minimax_h3"
_REF2VA_MODEL_TYPE = "minimax_h3_ref2va"
_FULL_MODEL_TYPE = "minimax_h3_full"
_REF2VA_FULL_MODEL_TYPE = "minimax_h3_ref2va_full"
_COMFY_REPO = "Comfy-Org/MiniMax-H3"
_COMFY_REVISION = "0543966fbdce5ba05709a8f2031c94bdba629b4a"
_OFFICIAL_REPO = "MiniMaxAI/MiniMax-H3"
_OFFICIAL_REVISION = "5d9b308a59ab12e67147f191e184baf704185bd1"
_DEEPBEEP_REPO = "DeepBeepMeep/MiniMax-H3"
_DEEPBEEP_REVISION = "fec7846aef352e58a1cfb699455e3d104281e68b"
_ASSETS_ROOT = "minimax_h3"

_TRANSFORMER = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
_REF2VA_TRANSFORMER = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
_TEXT_ENCODER_BF16 = "Qwen3-VL-32B-Instruct-layer50_bf16.safetensors"
_TEXT_ENCODER_INT8 = "Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors"
_TEXT_ENCODER_GGUF_Q2 = "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
_TEXT_ENCODER_GGUF_Q4 = "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

# H3 packs video, audio, and text into one unusually long transformer
# sequence.  At 480p / 10 seconds the token-wise activations alone need
# several gigabytes, so MMGP must not treat its model-weight safety cap as
# the entire available VRAM budget.  ``workingVRAM`` reserves this amount
# independently of the user's card size; MMGP streams more transformer
# blocks on smaller cards instead of starving the first denoising step.
_TRANSFORMER_WORKING_VRAM_MB = 10 * 1024

# H3's video VAE accepts 17*n+5 pixel frames. 345 is the final valid
# frame count at or below the official 15-second limit (14.375s at 24fps).
# First/Last may continue beyond that duration, but every individual model
# pass remains inside this native limit and reuses exactly one boundary frame.
_H3_MIN_FRAMES = 124
_H3_MAX_FRAMES = 345
_H3_SLIDING_WINDOW_DEFAULTS = {
    "window_min": _H3_MIN_FRAMES,
    "window_max": _H3_MAX_FRAMES,
    "window_step": 17,
    "window_default": _H3_MAX_FRAMES,
    "overlap_min": 1,
    "overlap_max": 1,
    "overlap_step": 0,
    "overlap_default": 1,
    "discard_last_frames": 0,
}


def _hf_url(repo_id: str, revision: str, *parts: str) -> str:
    path = "/".join(part.strip("/\\") for part in parts if part)
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{path}"


def _text_encoder_variants() -> dict[str, dict]:
    deepbeep_folder = "Qwen3-VL-32B-Instruct"
    return {
        "nvfp4_awq": {
            "name": "NVFP4 AWQ (Recommended)",
            "size_hint": (
                "~15.7 GB download · native acceleration requires an RTX 50-series "
                "GPU; RTX 40 and older use Maestro's proven compatibility fallback"
            ),
            "URLs": [
                _hf_url(
                    _COMFY_REPO,
                    _COMFY_REVISION,
                    "text_encoders",
                    _TEXT_ENCODER,
                )
            ],
        },
        "gguf_q2_k": {
            "name": "GGUF Q2_K (Lowest RAM)",
            "size_hint": "~8.5 GB download · lowest system-memory use",
            "URLs": [
                _hf_url(
                    _DEEPBEEP_REPO,
                    _DEEPBEEP_REVISION,
                    deepbeep_folder,
                    _TEXT_ENCODER_GGUF_Q2,
                )
            ],
        },
        "gguf_q4_k_m": {
            "name": "GGUF Q4_K_M",
            "size_hint": "~14.6 GB download · balanced quality and system-memory use",
            "URLs": [
                _hf_url(
                    _DEEPBEEP_REPO,
                    _DEEPBEEP_REVISION,
                    deepbeep_folder,
                    _TEXT_ENCODER_GGUF_Q4,
                )
            ],
        },
        "int8": {
            "name": "Quanto INT8",
            "size_hint": "~26.7 GB download · optional high-fidelity encoder with high system-memory use",
            "URLs": [
                _hf_url(
                    _DEEPBEEP_REPO,
                    _DEEPBEEP_REVISION,
                    deepbeep_folder,
                    _TEXT_ENCODER_INT8,
                )
            ],
        },
        "bf16": {
            "name": "BF16 (Maximum Fidelity)",
            "size_hint": "~51.5 GB download · maximum fidelity and very high system-memory use",
            "URLs": [
                _hf_url(
                    _DEEPBEEP_REPO,
                    _DEEPBEEP_REVISION,
                    deepbeep_folder,
                    _TEXT_ENCODER_BF16,
                )
            ],
        },
    }


def _recommend_text_encoder(hardware: dict | None, available=None) -> str:
    """Choose a proven encoder whose format fits system RAM."""

    choices = set(available or _text_encoder_variants())
    hardware = hardware or {}
    try:
        ram_gb = float(hardware.get("ram_gb") or 0)
    except (TypeError, ValueError):
        ram_gb = 0
    # The Comfy NVFP4-AWQ conditioner is Maestro's known-good H3 path.  RTX
    # 50 cards execute it natively; older NVIDIA cards use the compatibility
    # kernels successfully.  Prefer it whenever system RAM can hold it rather
    # than silently switching established users to a newly added encoder.
    if "nvfp4_awq" in choices and (hardware.get("supports_nvfp4") or ram_gb >= 24):
        return "nvfp4_awq"
    if ram_gb >= 56 and "int8" in choices:
        return "int8"
    if ram_gb >= 24 and "gguf_q4_k_m" in choices:
        return "gguf_q4_k_m"
    if "gguf_q2_k" in choices:
        return "gguf_q2_k"
    if "nvfp4_awq" in choices:
        return "nvfp4_awq"
    return next(iter(choices), "nvfp4_awq")


_H3_RESOLUTION_PRESETS = {
    "480p": {
        "label": "480p",
        "values": {
            "auto": "auto_480p",
            "16:9": "864x480",
            "9:16": "480x864",
            "1:1": "640x640",
            "4:3": "640x480",
            "3:4": "480x640",
        },
    },
    "540p": {
        "label": "540p",
        "values": {
            "auto": "auto_540p",
            "16:9": "960x544",
            "9:16": "544x960",
            "1:1": "736x736",
            "4:3": "736x544",
            "3:4": "544x736",
        },
    },
    # Keep the shared UI's stable 720p key, but label and map it to H3's
    # official 768px-short-edge native canvas.
    "720p": {
        "label": "768p",
        "values": {
            "auto": "auto_720p",
            "16:9": "1344x768",
            "9:16": "768x1344",
            "1:1": "768x768",
            "4:3": "1024x768",
            "3:4": "768x1024",
        },
    },
}
_H3_RESOLUTION_PRESET_ORDER = ["480p", "540p", "720p"]
_H3_AUTO_RESOLUTION_BUDGETS = {
    "auto": 1344 * 768,
    "auto_480p": 864 * 480,
    "auto_540p": 960 * 544,
    "auto_720p": 1344 * 768,
}
_H3_AUTO_RESOLUTION_FALLBACKS = {
    "auto": "1344x768",
    "auto_480p": "864x480",
    "auto_540p": "960x544",
    "auto_720p": "1344x768",
}

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
    ("736x544 (4:3)", "736x544"),
    ("544x736 (3:4)", "544x736"),
    ("736x736 (1:1)", "736x736"),
    ("864x480 (16:9 low VRAM)", "864x480"),
    ("480x864 (9:16 low VRAM)", "480x864"),
    ("640x480 (4:3 low VRAM)", "640x480"),
    ("480x640 (3:4 low VRAM)", "480x640"),
    ("640x640 (1:1 low VRAM)", "640x640"),
    ("608x352 (16:9 minimum)", "608x352"),
    ("352x608 (9:16 minimum)", "352x608"),
]

_LEGACY_RESOLUTION_ALIASES = {
    "848x480": "864x480",
    "480x848": "480x864",
    "672x672": "640x640",
    "832x608": "736x544",
    "608x832": "544x736",
    "1280x720": "1344x768",
    "1280x704": "1344x768",
    "720x1280": "768x1344",
    "704x1280": "768x1344",
    "1024x1024": "768x768",
    "1104x832": "1024x768",
    "832x1104": "768x1024",
    "1920x1088": "1344x768",
    "1088x1920": "768x1344",
}


def _normalize_h3_resolution(value) -> str:
    """Preserve requested orientation while snapping old presets to H3."""

    resolution = str(value or "864x480").strip().lower()
    if resolution in _H3_AUTO_RESOLUTION_BUDGETS:
        return resolution
    if resolution in _LEGACY_RESOLUTION_ALIASES:
        return _LEGACY_RESOLUTION_ALIASES[resolution]

    supported = [item for _, item in _RESOLUTIONS]
    if resolution in supported:
        return resolution
    try:
        width_text, height_text = resolution.split("x", 1)
        width, height = int(width_text), int(height_text)
        if width <= 0 or height <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return "864x480"

    orientation = 0 if width == height else (1 if width > height else -1)
    candidates = []
    for candidate in supported:
        candidate_width, candidate_height = (int(part) for part in candidate.split("x", 1))
        candidate_orientation = (
            0
            if candidate_width == candidate_height
            else (1 if candidate_width > candidate_height else -1)
        )
        if candidate_orientation == orientation:
            candidates.append((candidate, candidate_width, candidate_height))
    if not candidates:
        return "864x480"

    target_aspect = width / height
    target_area = width * height

    def score(item):
        _, candidate_width, candidate_height = item
        aspect_error = abs((candidate_width / candidate_height) - target_aspect) / target_aspect
        area_error = abs((candidate_width * candidate_height) - target_area) / target_area
        return aspect_error * 8 + area_error

    return min(candidates, key=score)[0]


class family_handler:
    @staticmethod
    def query_supported_types():
        return [
            _MODEL_TYPE,
            _FULL_MODEL_TYPE,
            _REF2VA_MODEL_TYPE,
            _REF2VA_FULL_MODEL_TYPE,
        ]

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
    def recommend_text_encoder(hardware, model_def=None):
        variants = (model_def or {}).get("minimax_h3_text_encoder_variants")
        return _recommend_text_encoder(hardware, variants)

    @staticmethod
    def query_model_def(base_model_type, model_def):
        omni_reference = base_model_type in {
            _REF2VA_MODEL_TYPE,
            _REF2VA_FULL_MODEL_TYPE,
        }
        full_checkpoint = base_model_type in {
            _FULL_MODEL_TYPE,
            _REF2VA_FULL_MODEL_TYPE,
        }
        text_encoder_variants = _text_encoder_variants()
        workflow_help = (
            "OMNI REFERENCES\n"
            "Use ordered image, video, and audio references to guide identity, "
            "appearance, motion, scenes, voices, or sound. The references guide "
            "a newly generated result rather than becoming fixed first/last frames. "
            "H3 also generates synchronized stereo audio. Omni uses one native "
            "model pass and is limited to 14.4 seconds."
            if omni_reference
            else
            "FIRST / LAST\n"
            "Generate from text alone, a first frame, a last frame, or both. H3 "
            "generates synchronized stereo audio, but this workflow does not accept "
            "reference audio. Longer videos continue through native 14.4-second "
            "windows using the prior window's last frame."
        )
        checkpoint_help = (
            "FULL 33B\n"
            "The larger original checkpoint uses more disk, RAM, and weight "
            "streaming. Choose it when you want the Full model or need an H3 "
            "Turbo / distilled LoRA."
            if full_checkpoint
            else
            "PRUNED 20B (RECOMMENDED)\n"
            "The lighter checkpoint has the same workflow controls with lower "
            "disk, RAM, and loading cost. H3 Turbo / distilled LoRAs are Full-only "
            "and are hidden while this model is selected."
        )
        result = {
            "dtype": "bf16",
            "fps": 24,
            # H3's video VAE accepts only 17*n+5 frames.  124 is the first
            # valid count at or above five seconds; 345 is the last at or
            # below fifteen seconds.
            "frames_minimum": _H3_MIN_FRAMES,
            "frames_steps": 17,
            "frames_maximum": _H3_MAX_FRAMES,
            "latent_size": 17,
            "frame_alignment_modulus": 17,
            "frame_alignment_remainder": 5,
            "frame_alignment_mode": "ceil",
            "sliding_window": not omni_reference,
            "video_continuation": not omni_reference,
            # The overall joined timeline need not itself lie on H3's
            # per-pass 17*n+5 grid. Keep the requested total exact, align
            # each pass independently, then trim only the final joined tail.
            "sliding_window_exact_total_frames": not omni_reference,
            "sliding_window_trim_to_requested": not omni_reference,
            "sliding_window_end_image_at_final": not omni_reference,
            # Director renders H3 as independent native-duration shots rather
            # than pretending it supports the rolling-window contract.
            "director_video_strategy": (
                "omni_reference" if omni_reference else "bounded_start_end"
            ),
            "director_audio_input_mode": (
                "reference_manifest" if omni_reference else "none"
            ),
            "director_reference_mode": (
                "omni_manifest" if omni_reference else "start_end"
            ),
            # Ref2VA consumes the user's character/location references
            # directly. FL2VA can render T2V or use generated start/end
            # frames, selected per Director project.
            "director_shot_image_support": (
                "direct_references" if omni_reference else "optional"
            ),
            "director_endpoint_continuity": not omni_reference,
            "director_trim_end_frames": False,
            "t2v_class": True,
            "i2v_class": not omni_reference,
            "image_prompt_types_allowed": "" if omni_reference else "TSE",
            "end_frames_always_enabled": not omni_reference,
            "returns_audio": True,
            # Ref2VA accepts audio through its ordered Omni manifest, not
            # through Wan's generic audio-guide input. Keep that capability
            # explicit so the UI can distinguish Audio In from Audio Out.
            "supports_reference_audio": omni_reference,
            "no_negative_prompt": True,
            "guidance_max_phases": 0,
            "visible_phases": 0,
            "compile": False,
            "resolutions": _RESOLUTIONS,
            "resolution_presets": _H3_RESOLUTION_PRESETS,
            "resolution_preset_order": _H3_RESOLUTION_PRESET_ORDER,
            "supports_auto_aspect": True,
            "auto_resolution_budgets": _H3_AUTO_RESOLUTION_BUDGETS,
            "auto_resolution_fallbacks": _H3_AUTO_RESOLUTION_FALLBACKS,
            "profiles_dir": ["minimax_h3"],
            "minimax_h3_assets_root": _ASSETS_ROOT,
            "text_encoder_folder": _ASSETS_ROOT,
            "text_encoder_quantization": "int8",
            "text_encoder_URLs": text_encoder_variants["nvfp4_awq"]["URLs"],
            "minimax_h3_text_encoder_default": "nvfp4_awq",
            "minimax_h3_text_encoder_variants": text_encoder_variants,
            "minimax_h3_full_checkpoint": full_checkpoint,
            "selector_help": f"{workflow_help}\n\n{checkpoint_help}",
            "lora_compatibility_note": (
                "H3 Turbo / distilled LoRAs are supported by this Full checkpoint."
                if full_checkpoint
                else
                "H3 Turbo / distilled LoRAs require an H3 Full model and are hidden for this Pruned model."
            ),
        }
        if omni_reference:
            result.update(
                {
                    "omni_reference": True,
                    "omni_reference_limits": {
                        "image": 9,
                        "video": 3,
                        "audio": 3,
                        "total": 12,
                    },
                    "omni_reference_detail_choices": [
                        ("Match output (recommended)", "match"),
                        ("Maximum reference detail", "max"),
                    ],
                    "omni_reference_detail_default": "match",
                }
            )
        else:
            result["sliding_window_defaults"] = dict(
                _H3_SLIDING_WINDOW_DEFAULTS
            )
        return result

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
            minimax_h3_text_encoder=kwargs.get(
                "minimax_h3_text_encoder",
                (model_def or {}).get("minimax_h3_text_encoder_default", "nvfp4_awq"),
            ),
        )
        pipe = {
            "transformer": model.transformer,
            # Profile the two Qwen towers independently. Text-only FL2VA
            # never needs the vision tower, while Ref2VA can release it
            # before the 50-layer language model runs. This mirrors WanGP's
            # H3 memory layout and avoids pinning both large components as a
            # single co-resident conditioner.
            "text_encoder": model.conditioner.language_model,
            "vision_encoder": model.conditioner.visual,
            "vae": model.vae,
            "audio_vae": model.audio_vae,
        }
        return model, {
            "pipe": pipe,
            "workingVRAM": {
                "transformer": _TRANSFORMER_WORKING_VRAM_MB,
            },
        }

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        omni_reference = base_model_type in {
            _REF2VA_MODEL_TYPE,
            _REF2VA_FULL_MODEL_TYPE,
        }
        ui_defaults.update(
            {
                "num_inference_steps": 20,
                "video_length": _H3_MIN_FRAMES,
                "resolution": "864x480",
                "guidance_scale": 1.0,
                "image_prompt_type": "",
                "sliding_window_size": (
                    _H3_MIN_FRAMES
                    if omni_reference
                    else _H3_MAX_FRAMES
                ),
                "sliding_window_overlap": 0 if omni_reference else 1,
                "sliding_window_discard_last_frames": 0,
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
        omni_reference = base_model_type in {
            _REF2VA_MODEL_TYPE,
            _REF2VA_FULL_MODEL_TYPE,
        }
        aligned_frames = align_num_frames(max(1, requested_frames))
        if omni_reference or requested_frames <= _H3_MAX_FRAMES + 1:
            ui_defaults["video_length"] = min(
                _H3_MAX_FRAMES,
                max(_H3_MIN_FRAMES, aligned_frames),
            )
        else:
            # A long First/Last setting is the joined output duration, not
            # one H3 pass, so preserve it for the sliding-window scheduler.
            ui_defaults["video_length"] = max(
                _H3_MIN_FRAMES,
                requested_frames,
            )

        try:
            requested_window = int(
                ui_defaults.get("sliding_window_size", _H3_MAX_FRAMES)
            )
        except (TypeError, ValueError):
            requested_window = _H3_MAX_FRAMES
        aligned_window = align_num_frames(max(1, requested_window))
        ui_defaults["sliding_window_size"] = (
            ui_defaults["video_length"]
            if omni_reference
            else min(
                _H3_MAX_FRAMES,
                max(_H3_MIN_FRAMES, aligned_window),
            )
        )
        ui_defaults["sliding_window_overlap"] = 0 if omni_reference else 1
        ui_defaults["sliding_window_discard_last_frames"] = 0
        ui_defaults["resolution"] = _normalize_h3_resolution(
            ui_defaults.get("resolution", "864x480")
        )
        ui_defaults["guidance_scale"] = 1.0

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        """Enforce H3's single-pass and continuation geometry server-side."""

        from .packing import align_num_frames

        omni_reference = base_model_type in {
            _REF2VA_MODEL_TYPE,
            _REF2VA_FULL_MODEL_TYPE,
        }
        try:
            requested_frames = int(inputs.get("video_length", _H3_MIN_FRAMES))
        except (TypeError, ValueError):
            requested_frames = _H3_MIN_FRAMES

        if omni_reference:
            inputs["video_length"] = min(
                _H3_MAX_FRAMES,
                max(_H3_MIN_FRAMES, align_num_frames(max(1, requested_frames))),
            )
            inputs["sliding_window_size"] = inputs["video_length"]
            inputs["sliding_window_overlap"] = 0
        else:
            if requested_frames <= _H3_MAX_FRAMES + 1:
                requested_frames = min(
                    _H3_MAX_FRAMES,
                    max(
                        _H3_MIN_FRAMES,
                        align_num_frames(max(1, requested_frames)),
                    ),
                )
            else:
                requested_frames = max(_H3_MIN_FRAMES, requested_frames)
            inputs["video_length"] = requested_frames

            try:
                requested_window = int(
                    inputs.get("sliding_window_size", _H3_MAX_FRAMES)
                )
            except (TypeError, ValueError):
                requested_window = _H3_MAX_FRAMES
            inputs["sliding_window_size"] = min(
                _H3_MAX_FRAMES,
                max(
                    _H3_MIN_FRAMES,
                    align_num_frames(max(1, requested_window)),
                ),
            )
            inputs["sliding_window_overlap"] = 1

        inputs["sliding_window_discard_last_frames"] = 0
        inputs["sliding_window_overlap_noise"] = 0
        inputs["sliding_window_color_correction_strength"] = 0
        return None
