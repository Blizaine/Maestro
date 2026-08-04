"""Maestro family handler for MiniMax H3 Base.

Two checkpoints are exposed.  FL2VA is the unified one: text-to-video,
image-to-video, first/last-frame video, and continuation, all with native
stereo audio.  Ref2VA conditions on material that is not a frame of the
output -- reference stills carrying an identity or a subject, a reference
clip carrying a look or a motion, and audio references carrying a voice,
including the reference clip's own soundtrack.

One reference clip, not two.  H3 accepts a second, but Maestro has a single
``video_guide`` input and no ``video_guide2``; adding one means new
attachment plumbing shared by every model, so the second slot is left unbuilt
rather than offered as a control that cannot receive a file.
"""

from __future__ import annotations

import os

import torch

from .prompt_enhancer import (
    FL2VA_IMAGE_SYSTEM_PROMPT,
    FL2VA_PROMPT_INFOS,
    FL2VA_TEXT_SYSTEM_PROMPT,
    REF2VA_IMAGE_SYSTEM_PROMPT,
    REF2VA_PROMPT_INFOS,
    REF2VA_TEXT_SYSTEM_PROMPT,
)


_MODEL_TYPE = "minimax_h3"
_MODEL_TYPE_REF2VA = "minimax_h3_ref2va"
_COMFY_REPO = "Comfy-Org/MiniMax-H3"
_COMFY_REVISION = "0543966fbdce5ba05709a8f2031c94bdba629b4a"
_OFFICIAL_REPO = "MiniMaxAI/MiniMax-H3"
_OFFICIAL_REVISION = "5d9b308a59ab12e67147f191e184baf704185bd1"
_ASSETS_ROOT = "minimax_h3"

_TRANSFORMER = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

# H3 packs video, audio, and text into one unusually long transformer
# sequence.  At 480p / 10 seconds the token-wise activations alone need
# several gigabytes, so MMGP must not treat its model-weight safety cap as
# the entire available VRAM budget.  ``workingVRAM`` reserves this amount
# independently of the user's card size; MMGP streams more transformer
# blocks on smaller cards instead of starving the first denoising step.
#
# Sized for the largest canvas the model will accept rather than for 480p.
# Attention holds query, key and value for the whole packed sequence at
# once, each of them ``rows * heads * head_dim`` -- about 1 GB apiece per
# 80k rows -- and the residual stream alongside them.  A 480p-sized
# reservation left MMGP holding ~26 GB of weights on a 31 GB card, which is
# short by roughly one of those tensors, so the first block of the first
# step died allocating ``value``.  This is a worst-case figure because
# ``load_model`` is per-model, not per-generation: it cannot know the
# resolution, so it must cover the largest.  The cost is more block
# streaming at small canvases, which is slower but does not fail.
_TRANSFORMER_WORKING_VRAM_MB = 16 * 1024


# The shortest edge that still counts as "meant to be native". 640 is a canvas H3 lists in its own right
# (1152x640), so the band starts above it: a request for 640 is a request for 640, while 704 or 720 is a
# generic preset aiming at the same scale as native and falling short.
_NATIVE_LIFT_FLOOR = 672


def _snap_resolution(resolution):
    """Snap a requested canvas onto the grid H3 can actually sample, keeping its size and aspect.

    H3 needs both axes on a multiple of 32. The UI offers generic presets -- 848x480, 1280x720, 1920x1080 --
    and none of those are multiples of 32 on both axes, so a membership test against a curated list rejects
    almost all of them.

    Rejecting used to mean falling back to 864x480, which is why every preset except 540p silently produced
    a 480p render. Snapping keeps the resolution that was asked for: 1280x720 becomes 1280x704, 1920x1080
    becomes 1920x1088.

    Only the 32px grid is enforced, because that is the only thing `generate` actually requires. H3 defines
    a 768*1344 figure, but it belongs to `resolve_canvas_size`, which resolves an aspect ratio into a
    default canvas for reference material -- it is not a ceiling on the output, and imposing it here would
    cap 1080p for no reason. What limits resolution in practice is VRAM.

    Anything that is not a concrete WxH is returned untouched. The "auto" aspect resolves to sentinels like
    `auto_720p`, which are resolved downstream from the source material -- replacing one with a fixed canvas
    would be the same bug wearing a different hat.
    """
    from .packing import MINIMAX_H3_CANVAS_MULTIPLE, MINIMAX_H3_SHORT_EDGE

    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    try:
        raw_width, raw_height = (int(part) for part in str(resolution).lower().split("x", 1))
    except (TypeError, ValueError):
        return resolution
    if raw_width <= 0 or raw_height <= 0:
        return resolution

    # A request that lands just under H3's native scale is lifted onto it, keeping its aspect ratio. The
    # model's short edge is 768 -- `resolve_canvas_size` targets it and upstream's own list calls 1344x768
    # "16:9 native" -- and a generic 720p preset rounds to 704, close enough to native to be clearly meant
    # as it, but under the scale the model was trained at, which costs sharpness for nothing.
    #
    # Deliberately narrow, so smaller presets stay where they were put. The long edge must already reach
    # native, which is what separates "a wide canvas falling short on its short edge" from "a small square":
    # 540p's 736x736 and 480p's 672x672 have short edges inside the band but are simply small requests, and
    # a plain short-edge test would inflate both.
    short_edge, long_edge = min(raw_width, raw_height), max(raw_width, raw_height)
    if _NATIVE_LIFT_FLOOR <= short_edge < MINIMAX_H3_SHORT_EDGE and long_edge >= MINIMAX_H3_SHORT_EDGE:
        scale = MINIMAX_H3_SHORT_EDGE / short_edge
        raw_width, raw_height = raw_width * scale, raw_height * scale

    width = max(multiple, int(round(raw_width / multiple)) * multiple)
    height = max(multiple, int(round(raw_height / multiple)) * multiple)
    return f"{width}x{height}"


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
        return [_MODEL_TYPE, _MODEL_TYPE_REF2VA]

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
        reference_mode = base_model_type == _MODEL_TYPE_REF2VA
        definition = {
            "dtype": "bf16",
            "fps": 24,
            # H3's video VAE accepts only 17*n+5 frames.  124 is the first
            # valid count at or above five seconds; 345 is the last at or
            # below fifteen seconds.
            # H3's prompt is one structured block (integrated_multimodal_
            # description / overall_soundscape / non_diegetic_music separated
            # by blank lines). Without this Maestro splits it on newlines and
            # runs each field as its own generation.
            "single_block_prompt": True,
            "preserve_empty_prompt_lines": True,
            "frames_minimum": 124,
            "frames_steps": 17,
            "frames_maximum": 345,
            "latent_size": 17,
            "frame_alignment_modulus": 17,
            "frame_alignment_remainder": 5,
            "frame_alignment_mode": "ceil",
            "sliding_window": False,
            "t2v_class": True,
            "i2v_class": not reference_mode,
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
            # H3 does not read a free-form prompt: it reads a structured, field-by-field block with its own
            # dialogue markup. A generic enhancer rewrites that into prose the model cannot parse, so the two
            # tasks get their own instructions. Ref2VA's are longer because its prompt has six sections, not three.
            "prompt_infos": REF2VA_PROMPT_INFOS if reference_mode else FL2VA_PROMPT_INFOS,
            "prompt_enhancer_button_label": "Write H3 Prompt",
            "prompt_enhancer_def": {
                "selection": ["T", "TI"],
                "labels": {
                    "T": "Write an H3 Reference Prompt from Text"
                    if reference_mode
                    else "Write an H3 Prompt from Text",
                    "TI": "Write an H3 Reference Prompt from Text + First Reference Image"
                    if reference_mode
                    else "Write an H3 Prompt from Text + Start Image",
                },
                "default": "",
            },
            "text_prompt_enhancer_instructions": (
                REF2VA_TEXT_SYSTEM_PROMPT if reference_mode else FL2VA_TEXT_SYSTEM_PROMPT
            ),
            "video_prompt_enhancer_instructions": (
                REF2VA_IMAGE_SYSTEM_PROMPT if reference_mode else FL2VA_IMAGE_SYSTEM_PROMPT
            ),
            "text_prompt_enhancer_max_tokens": 2048 if reference_mode else 1024,
            "video_prompt_enhancer_max_tokens": 2048 if reference_mode else 1024,
        }
        if reference_mode:
            # Ref2VA conditions on material that is not on the output timeline, so it takes no start/end frame.
            # Reference stills are passed as image refs and left alone: no background removal, no resizing onto the
            # target canvas -- each keeps its own framing, which is the whole point of a reference.
            definition.update(
                {
                    "image_prompt_types_allowed": "T",
                    "reference_image_enabled": True,
                    "return_image_refs_tensor": False,
                    "no_background_removal": True,
                    "no_processing_on_last_images_refs": 9,
                    "image_ref_choices": {
                        "choices": [
                            ("Generate without Reference Images", ""),
                            ("Use Reference Images", "I"),
                        ],
                        "letters_filter": "I",
                        "default": "",
                        "label": "Reference Images",
                    },
                    # A reference clip is decoded by the model from its source path, not taken from the
                    # guide pipeline -- see `reference_video_source_path` and `_decode_reference_video`.
                    "reference_video_source_path": True,
                    "guide_custom_choices": {
                        "choices": [
                            ("Generate without a Reference Video", ""),
                            ("Use One Reference Video", "V"),
                        ],
                        "letters_filter": "V",
                        "default": "",
                        "label": "Reference Video",
                    },
                    "video_guide_label": "Reference Video",
                    "any_audio_prompt": True,
                    "audio_prompt_choices": True,
                    "audio_guide_label": "Audio Reference 1",
                    "audio_guide2_label": "Audio Reference 2",
                    "audio_prompt_type_sources": {
                        "selection": ["", "A", "AB", "K"],
                        "labels": {
                            "": "Generate without an Audio Reference",
                            "A": "Use One Audio Reference",
                            "AB": "Use Two Audio References",
                            "K": "Use the Reference Video's Soundtrack",
                        },
                        "letters_filter": "ABK",
                        "label": "Audio References",
                        "show_label": True,
                        "default": "",
                    },
                    "video_length_not_limited_by_audio": True,
                }
            )
        else:
            definition.update(
                {
                    # "V" enables Studio's Extend: H3 cannot generate past 15s in one pass, so a continuation
                    # starts a fresh clip from the last frame of the previous one rather than sliding a window
                    # over shared latents.
                    "image_prompt_types_allowed": "TSEV",
                    "video_continuation": True,
                    "end_frames_always_enabled": True,
                }
            )
        return definition

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
    def get_rgb_factors(base_model_type):
        # Without this wgp's preview helper returns None and the progress
        # pane stays empty for the whole generation.
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("minimax_h3")

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
        return model, {
            "pipe": pipe,
            "workingVRAM": {
                "transformer": _TRANSFORMER_WORKING_VRAM_MB,
            },
        }

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
        if base_model_type == _MODEL_TYPE_REF2VA:
            # Both selectors start empty: Ref2VA generates from the prompt alone until references are added.
            ui_defaults.setdefault("video_prompt_type", "")
            ui_defaults.setdefault("audio_prompt_type", "")

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        # The UI's resolution presets are generic and mostly land off H3's 32px grid, so snap here as well
        # as in fix_settings: this is the path a resolution takes when it is chosen rather than loaded.
        resolution = inputs.get("resolution")
        if resolution:
            inputs["resolution"] = _snap_resolution(resolution)

        # Ref2VA's limits are cheap to check here and expensive to discover after a model has been loaded and a
        # generation has started, which is where an over-long reference otherwise surfaces.
        if base_model_type != _MODEL_TYPE_REF2VA:
            return None

        if len(inputs.get("image_refs") or []) > 9:
            return "MiniMax H3 Ref2VA accepts at most 9 reference images"

        if "V" in (inputs.get("video_prompt_type") or ""):
            reference_video = inputs.get("video_guide")
            if reference_video is None:
                return "A Reference Video is selected but no file was provided"
            from shared.utils.utils import get_video_info

            try:
                fps, _, _, frames = get_video_info(reference_video)
                duration = frames / fps
            except Exception as error:
                return f"Unable to read the Reference Video: {error}"
            if not 2.0 <= duration <= 15.0:
                return f"The Reference Video must be between 2 and 15 seconds long (found {duration:.2f}s)"

        audio_prompt_type = inputs.get("audio_prompt_type") or ""
        if "K" in audio_prompt_type:
            # Caught here because the alternative is discovering it after the model has loaded: a clip with
            # no audio track would simply contribute no reference, and the generation would look like the
            # soundtrack option had been ignored.
            if "V" not in (inputs.get("video_prompt_type") or "") or not inputs.get("video_guide"):
                return "Using the Reference Video's soundtrack requires a Reference Video"
            from shared.utils.audio_video import extract_audio_tracks

            try:
                if extract_audio_tracks(inputs["video_guide"], query_only=True) == 0:
                    return "The Reference Video has no audio track to use as a soundtrack reference"
            except Exception as error:
                return f"Unable to inspect the Reference Video's soundtrack: {error}"

        references = [inputs.get("audio_guide")] if "A" in audio_prompt_type else []
        if "B" in audio_prompt_type:
            references.append(inputs.get("audio_guide2"))

        import librosa

        for index, reference in enumerate(references, 1):
            if reference is None:
                return f"Audio Reference {index} is selected but no file was provided"
            try:
                duration = float(librosa.get_duration(path=os.fspath(reference)))
            except Exception as error:
                return f"Unable to read Audio Reference {index}: {error}"
            if not 2.0 <= duration <= 15.0:
                return (
                    f"Audio Reference {index} must be between 2 and 15 seconds long (found {duration:.2f}s)"
                )
        return None

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
        ui_defaults["resolution"] = _snap_resolution(ui_defaults.get("resolution", "864x480"))
        ui_defaults["guidance_scale"] = 1.0
