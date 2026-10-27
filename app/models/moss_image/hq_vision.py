# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Qwen3.5 vision conditioning used by the community MOSS HQ v5 adapter."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from accelerate import init_empty_weights
from mmgp import offload

from shared.prompt_enhancer.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from shared.prompt_enhancer.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Model,
    Qwen3_5VisionModel,
    Qwen3_5VisionRotaryEmbedding,
)
from shared.prompt_enhancer.qwen35_vl import _load_qwen35_image_processor
from shared.utils import files_locator as fl

from .runtime import runtime


DEFAULT_HQ_VISION_FOLDER = "moss_image_hq_vision"
DEFAULT_HQ_VISION_FILE = os.path.join(
    "Qwen3_5_9B_Abliterated", "Qwen3.5-9B-vision_bf16.safetensors"
)
DEFAULT_HQ_VISION_PIXELS = 512 * 512
HQ_VISION_LAYOUT = "moss-hq-v5-ai-toolkit-picture-prefix-v1"
MAX_HQ_TEXT_TOKENS = 4096


class Qwen3_5VisionPositionIndexer:
    """Expose the bundled Qwen 3-D RoPE helpers without constructing its LLM."""

    get_vision_position_ids = Qwen3_5Model.get_vision_position_ids
    get_rope_index = Qwen3_5Model.get_rope_index

    def __init__(self, config):
        self.config = config


def restore_vision_rotary_buffers(vision_encoder):
    """Rebuild Qwen vision RoPE frequencies that are intentionally absent from checkpoints."""
    for module in vision_encoder.modules():
        if isinstance(module, Qwen3_5VisionRotaryEmbedding):
            module.inv_freq = 1.0 / (
                module.theta
                ** (torch.arange(0, module.dim, 2, dtype=torch.float32) / module.dim)
            )
            module._lock_dtype = torch.float32


def _required_asset(folder: str, relative_path: str, label: str) -> str:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"MOSS Image HQ vision {label} must be a relative asset path.")
    candidate = os.path.join(folder, str(relative))
    located = candidate if os.path.isfile(candidate) else fl.locate_file(candidate, error_if_none=False)
    if located is None:
        raise FileNotFoundError(
            f"MOSS Image Edit HQ v5 is missing its Qwen3.5 vision {label}: {candidate}. "
            "Download the HQ vision assets before loading this model."
        )
    return os.path.abspath(located)


def load_hq_vision_components(model_def):
    """Load the pinned vision tower and its local processor, never a replacement LLM."""
    folder = str(model_def.get("moss_edit_vision_folder") or DEFAULT_HQ_VISION_FOLDER).strip()
    filename = str(model_def.get("moss_edit_vision_file") or DEFAULT_HQ_VISION_FILE).strip()
    if not folder:
        raise ValueError("MOSS Image Edit HQ v5 requires a Qwen3.5 vision asset folder.")

    config_path = _required_asset(folder, "config.json", "configuration")
    processor_path = _required_asset(folder, "preprocessor_config.json", "image processor configuration")
    vision_path = _required_asset(folder, filename, "checkpoint")
    asset_folder = str(Path(config_path).parent)
    if str(Path(processor_path).parent) != asset_folder:
        raise ValueError("MOSS Image HQ vision configuration and image processor must share one asset folder.")

    full_config = Qwen3_5Config.from_json_file(config_path)
    if full_config.vision_config is None:
        raise ValueError("MOSS Image HQ vision configuration has no vision_config.")
    full_config.vision_config._attn_implementation = "sdpa"

    with init_empty_weights(include_buffers=True):
        vision_encoder = Qwen3_5VisionModel(full_config.vision_config)
    restore_vision_rotary_buffers(vision_encoder)
    print(f"[MOSS Image Edit HQ] Loading Qwen vision checkpoint: {vision_path}")
    offload.load_model_data(
        vision_encoder,
        [vision_path],
        modelPrefix="model.visual",
        writable_tensors=False,
        default_dtype=torch.bfloat16,
    )
    vision_encoder.eval().requires_grad_(False)

    image_processor = _load_qwen35_image_processor(asset_folder)
    token_ids = {
        "image": int(full_config.image_token_id),
        "vision_start": int(full_config.vision_start_token_id),
        "vision_end": int(full_config.vision_end_token_id),
    }
    return vision_encoder, image_processor, full_config, token_ids


def _module_device_dtype(module):
    device = runtime.execution_device(module)
    for tensor in (*module.parameters(), *module.buffers()):
        if not tensor.is_meta:
            return device, tensor.dtype
    raise RuntimeError("MOSS Image HQ vision encoder has no loaded parameters.")


@torch.inference_mode()
def encode_vision_references(vision_encoder, image_processor, images, max_pixels=DEFAULT_HQ_VISION_PIXELS):
    """Encode ordered RGB references into Qwen visual tokens and patch grids."""
    if not images:
        raise ValueError("MOSS Image Edit HQ v5 requires one to four reference images.")
    if isinstance(max_pixels, bool):
        raise ValueError("MOSS Image HQ vision pixel budget must be a positive integer.")
    try:
        normalized_max_pixels = int(max_pixels)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("MOSS Image HQ vision pixel budget must be a positive integer.") from error
    if normalized_max_pixels != max_pixels or normalized_max_pixels <= 0:
        raise ValueError("MOSS Image HQ vision pixel budget must be a positive integer.")

    prepared = [image.convert("RGB") for image in images]
    processor_size = getattr(image_processor, "size", {}) or {}
    try:
        min_pixels = int(processor_size.get("shortest_edge", 56 * 56))
    except (TypeError, ValueError, AttributeError):
        min_pixels = 56 * 56
    min_pixels = min(min_pixels, normalized_max_pixels)
    model_inputs = image_processor(
        images=prepared,
        return_tensors="pt",
        size={"shortest_edge": min_pixels, "longest_edge": normalized_max_pixels},
    )
    pixel_values = model_inputs.get("pixel_values")
    image_grid_thw = model_inputs.get("image_grid_thw")
    if not torch.is_tensor(pixel_values) or not torch.is_tensor(image_grid_thw):
        raise ValueError("MOSS Image HQ image processor did not return pixel_values and image_grid_thw.")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3 or image_grid_thw.shape[0] != len(images):
        raise ValueError("MOSS Image HQ image processor returned an invalid ordered image grid.")
    patch_size = int(getattr(image_processor, "patch_size", 16))
    if patch_size <= 0 or not bool(torch.all(image_grid_thw[:, 0] == 1)):
        raise ValueError("MOSS Image HQ image processor returned an unsupported still-image patch grid.")
    patch_areas = image_grid_thw[:, 1] * image_grid_thw[:, 2] * patch_size * patch_size
    if bool(torch.any(patch_areas > normalized_max_pixels)):
        raise ValueError(
            "MOSS Image HQ image processor exceeded the configured per-reference vision pixel budget."
        )

    device, dtype = _module_device_dtype(vision_encoder)
    output = vision_encoder(
        pixel_values.to(device=device, dtype=dtype),
        grid_thw=image_grid_thw.to(device=device),
        return_dict=True,
    )
    features = getattr(output, "pooler_output", None)
    if not torch.is_tensor(features):
        raise ValueError("MOSS Image HQ vision encoder did not return pooled image features.")
    merge_size = int(vision_encoder.spatial_merge_size)
    feature_counts = (image_grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
    if sum(feature_counts) != features.shape[0] or any(count <= 0 for count in feature_counts):
        raise ValueError("MOSS Image HQ vision feature count does not match the ordered image grids.")
    return tuple(features.split(feature_counts)), image_grid_thw.to(dtype=torch.long)


def _special_token_id(tokenizer, token: str, expected: int) -> None:
    actual = tokenizer.convert_tokens_to_ids(token)
    try:
        actual = int(actual)
    except (TypeError, ValueError):
        actual = -1
    if actual != int(expected):
        raise ValueError(
            f"MOSS Image HQ requires tokenizer token {token} to have id {expected}; got {actual}."
        )


def _prompt_content(prompt: str, image_counts) -> str:
    prefixes = []
    for index, count in enumerate(image_counts, start=1):
        prefixes.append(
            f"Picture {index}: <|vision_start|>{'<|image_pad|>' * int(count)}<|vision_end|>"
        )
    return "".join(prefixes) + str(prompt or "")


def encode_prompt_with_vision(
    encoder,
    tokenizer,
    prompt,
    image_features,
    image_grid_thw,
    vision_config,
    token_ids,
    position_indexer,
):
    """Encode one CFG prompt with ordered vision features and Qwen 3-D RoPE."""
    if not image_features:
        raise ValueError("MOSS Image Edit HQ v5 requires visual features for every reference image.")
    if len(image_features) != int(image_grid_thw.shape[0]):
        raise ValueError("MOSS Image HQ visual features and reference grids have different lengths.")

    merge_size = int(vision_config.vision_config.spatial_merge_size)
    image_counts = [int(value) for value in (image_grid_thw.prod(-1) // (merge_size * merge_size)).tolist()]
    if any(int(features.shape[0]) != count for features, count in zip(image_features, image_counts)):
        raise ValueError("MOSS Image HQ visual feature lengths do not match their reference grids.")

    image_token_id = int(token_ids["image"])
    _special_token_id(tokenizer, "<|image_pad|>", image_token_id)
    _special_token_id(tokenizer, "<|vision_start|>", int(token_ids["vision_start"]))
    _special_token_id(tokenizer, "<|vision_end|>", int(token_ids["vision_end"]))

    content = _prompt_content(prompt, image_counts)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt", truncation=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("MOSS Image HQ expects one unpadded prompt per text-encoder call.")
    if input_ids.shape[1] > MAX_HQ_TEXT_TOKENS:
        raise ValueError(
            f"MOSS Image HQ prompt and reference tokens exceed {MAX_HQ_TEXT_TOKENS} tokens. "
            "Use fewer references or a shorter instruction."
        )

    device = runtime.execution_device(encoder)
    input_ids = input_ids.to(device=device, dtype=torch.long)
    image_mask = input_ids.eq(image_token_id)
    total_image_tokens = sum(image_counts)
    if int(image_mask.sum().item()) != total_image_tokens:
        raise ValueError(
            "MOSS Image HQ image placeholders do not match the Qwen vision feature lengths."
        )

    ordered_features = torch.cat(list(image_features), dim=0)
    mm_token_type_ids = image_mask.to(dtype=torch.int32)
    image_grid_thw = image_grid_thw.to(device=device, dtype=torch.long)
    position_ids, _ = position_indexer.get_rope_index(
        input_ids,
        mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=torch.ones_like(input_ids),
    )
    if tuple(position_ids.shape) != (3, 1, input_ids.shape[1]):
        raise ValueError("MOSS Image HQ Qwen position helper returned an invalid 3-D position layout.")

    from .progress import text_encoding_progress

    with text_encoding_progress(encoder.layers):
        return encoder(
            input_ids,
            position_ids=position_ids,
            image_features=ordered_features,
            image_token_id=image_token_id,
        )


def vision_prompt_cache_key(prompt, images, max_pixels=DEFAULT_HQ_VISION_PIXELS):
    """Key cached HQ prompt states by text, vision layout, budget and ordered pixels."""
    references = []
    for image in images:
        rgb = image.convert("RGB")
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        references.append((tuple(rgb.size), str(image.mode), str(image.format or ""), digest))
    return (
        HQ_VISION_LAYOUT,
        int(max_pixels),
        str(prompt or ""),
        tuple(references),
    )

