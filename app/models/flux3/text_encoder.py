"""Qwen3-VL text features used by the BFL FLUX 3 reference; see LICENSE."""

import math
from collections.abc import Callable

import torch
from torch import Tensor


OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
MAX_LENGTH = 4096
PAD_MULTIPLE = 80
FEATURE_WIDTH = 20480


def encode_prompt(model, processor, prompt: str, device: torch.device | str,
                  check_cancel: Callable[[], None] | None = None) -> tuple[Tensor, Tensor]:
    """Encode one prompt as padded Qwen hidden features plus FLUX text IDs.

    The causal LM head is intentionally not called: FLUX consumes the selected
    Qwen hidden layers directly, so materializing vocabulary logits is wasted
    memory. ``model`` is the bare ``Qwen3VLModel`` loaded from checkpoint keys
    under the official ``model.`` prefix.
    """
    device = torch.device(device)
    if check_cancel is not None:
        check_cancel()
    text = processor.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    tokenizer = processor.tokenizer
    real_length = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
    )["input_ids"].shape[1]
    length = min(math.ceil(real_length / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_LENGTH)
    inputs = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=length
    )
    attention_mask = inputs["attention_mask"].to(device)
    positions = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)
    input_ids = inputs["input_ids"].to(device)
    position_ids = positions[None].expand(4, -1, -1)
    if check_cancel is not None:
        check_cancel()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    if check_cancel is not None:
        check_cancel()
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None or max(OUTPUT_LAYERS) >= len(hidden_states):
        raise RuntimeError("Qwen3-VL did not return the hidden layers required by FLUX 3.")
    stacked = torch.stack([hidden_states[layer] for layer in OUTPUT_LAYERS], dim=1)
    tokens = stacked.permute(0, 2, 1, 3).reshape(length, -1).to(torch.bfloat16)
    if tokens.shape != (length, FEATURE_WIDTH):
        raise RuntimeError(
            f"Qwen3-VL produced text features with shape {tuple(tokens.shape)}; "
            f"FLUX 3 requires ({length}, {FEATURE_WIDTH})."
        )
    sequence_positions = torch.arange(length, device=device)
    ids = torch.stack([torch.zeros_like(sequence_positions)] * 3 + [sequence_positions], dim=-1)
    return tokens, ids
