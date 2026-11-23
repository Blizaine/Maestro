"""Adapter-format conversion for the upstream LongCat Avatar v1.6 LoRAs."""

from __future__ import annotations

import torch

from models.moss_image.lora_mapping import convert_lora_keys


_UPSTREAM_LORA_PREFIX = "lora___lorahyphen___"


def preprocess_longcat_avatar_v1_6_lora(
    state_dict,
    module_names,
    *,
    split_linear_modules_map=None,
):
    """Convert upstream Avatar v1.6 LoRA modules to MMGP's adapter names.

    Other LongCat LoRA formats pass through unchanged. The v1.6 checkpoints
    store module names in escaped state-dict prefixes, save native alpha scale
    buffers, and split fused projection up weights into parallel blocks.
    """

    if not any(key.startswith(_UPSTREAM_LORA_PREFIX) for key in state_dict):
        return state_dict

    def decode_module_name(prefix):
        return prefix.replace(_UPSTREAM_LORA_PREFIX, "", 1).replace(
            "___lorahyphen___", "."
        )

    def block_index(key):
        return int(key.split(".lora_up.blocks.", 1)[1].split(".", 1)[0])

    converted = {}
    prefixes = sorted(
        key.rsplit(".lora_down.weight", 1)[0]
        for key in state_dict
        if key.startswith(_UPSTREAM_LORA_PREFIX)
        and key.endswith(".lora_down.weight")
    )
    for prefix in prefixes:
        module_name = decode_module_name(prefix)
        down = state_dict.pop(f"{prefix}.lora_down.weight")
        converted[f"{module_name}.lora_down.weight"] = down

        alpha_scale = state_dict.pop(f"{prefix}.alpha_scale", None)
        if alpha_scale is not None:
            alpha_scale = (
                float(alpha_scale.item())
                if torch.is_tensor(alpha_scale)
                else float(alpha_scale)
            )
            # MMGP uses alpha / rank; this reconstructs the exact native
            # LongCat alpha_scale when its combined rank spans fused Q/K/V.
            converted[f"{module_name}.alpha"] = torch.tensor(
                alpha_scale * down.shape[0], dtype=torch.float32
            )

        up_key = f"{prefix}.lora_up.weight"
        if up_key in state_dict:
            converted[f"{module_name}.lora_up.weight"] = state_dict.pop(up_key)
            continue

        up_block_keys = sorted(
            (
                key
                for key in state_dict
                if key.startswith(f"{prefix}.lora_up.blocks.")
                and key.endswith(".weight")
            ),
            key=block_index,
        )
        if not up_block_keys:
            raise ValueError(f"Missing LongCat LoRA up weights for {module_name}.")
        total_rank = sum(state_dict[key].shape[1] for key in up_block_keys)
        if down.ndim != 2 or down.shape[0] != total_rank:
            raise ValueError(
                f"LongCat LoRA rank mismatch for {module_name}: "
                f"down={tuple(down.shape)} up rank={total_rank}"
            )
        up = down.new_zeros(
            (sum(state_dict[key].shape[0] for key in up_block_keys), down.shape[0])
        )
        row_start = 0
        col_start = 0
        for key in up_block_keys:
            block = state_dict.pop(key)
            if block.ndim != 2:
                raise ValueError(f"Invalid LongCat LoRA up block for {module_name}.")
            rows, rank = block.shape
            up[
                row_start : row_start + rows,
                col_start : col_start + rank,
            ] = block if block.dtype == up.dtype else block.to(up.dtype)
            row_start += rows
            col_start += rank
        converted[f"{module_name}.lora_up.weight"] = up

    for key, value in state_dict.items():
        if key.startswith(_UPSTREAM_LORA_PREFIX):
            raise ValueError(f"Unsupported LongCat Avatar v1.6 LoRA key: {key}")
        converted[key] = value

    return convert_lora_keys(
        converted,
        module_names,
        split_linear_modules_map=split_linear_modules_map,
    )


def make_longcat_avatar_v1_6_lora_preprocessor(transformer):
    """Build the ``preprocess_loras(model_type, state_dict)`` WGP hook."""

    module_names = dict(transformer.named_modules())
    split_linear_modules_map = getattr(transformer, "split_linear_modules_map", None)

    def preprocess_loras(_model_type, state_dict):
        return preprocess_longcat_avatar_v1_6_lora(
            state_dict,
            module_names,
            split_linear_modules_map=split_linear_modules_map,
        )

    return preprocess_loras
