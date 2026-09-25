"""Memory planning for Qwen Image 2.1's encoder and reference prefix."""

import itertools

import torch
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


ENCODER_ATTENTION = "maestro_qwen21_sdpa"
GIB = 1024 ** 3


def encoder_attention(module, query, key, value, attention_mask, dropout=0.0,
                      scaling=None, is_causal=None, **kwargs):
    """Keep SDPA's efficient kernels eligible on builds without native Flash.

    Transformers enables native GQA for unpadded Qwen3-VL prompts. Torch's
    efficient attention cannot use that flag; Windows builds without native
    Flash then materialize quadratic FP32 attention scores. Expanding only
    the K/V heads is linear in sequence length and computes the same attention.
    """
    groups = query.shape[1] // key.shape[1]
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, :key.shape[-2]]
    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        scale=scaling, is_causal=is_causal,
    )
    return output.transpose(1, 2).contiguous(), None


def configure_encoder_attention(model):
    # Register a separate name: never replace SDPA globally for other models.
    # The mask registration is equally important: an unknown mask backend
    # would silently discard padding and causal masks in Transformers.
    ALL_ATTENTION_FUNCTIONS.register(ENCODER_ATTENTION, encoder_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(ENCODER_ATTENTION, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    model.set_attn_implementation(ENCODER_ATTENTION)


def model_storage_bytes(model):
    """Count physical weights, including INT8 data/scales rather than FP32 views."""
    total = 0
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        getter = getattr(tensor, "get_quantized_subtensors", None)
        parts = getter() if getter is not None else None
        if isinstance(parts, dict):
            parts = parts.items()
        tensors = [part for _, part in parts if torch.is_tensor(part)] if parts else [tensor]
        total += sum(part.numel() * part.element_size() for part in tensors)
    return total


def reference_cache_plan(transformer, prompt_embeds, image_mask, reference_rows,
                         target_rows, device, negative_prompt_embeds=None,
                         negative_image_mask=None):
    """Return cache bytes and a conservative budget before any cache allocation.

    MMGP manages weights, not the per-layer K/V tensors. Count the actual
    interleaved text/reference prefix, both CFG branches, and expanded batch.
    Reserve full transformer weights plus workspace even if another stage is
    currently resident. This may forgo a cache on small GPUs but never changes
    references, image resolution, or the model's attention semantics.
    """
    width = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    layers = len(transformer.transformer_blocks)
    batch = prompt_embeds.shape[0]
    itemsize = prompt_embeds.element_size()
    prefix_rows = reference_rows + int((~image_mask[0].bool()).sum().item())
    cache_rows = prefix_rows
    largest_prefix = prefix_rows
    if negative_prompt_embeds is not None:
        negative_rows = reference_rows + int((~negative_image_mask[0].bool()).sum().item())
        cache_rows += negative_rows
        largest_prefix = max(largest_prefix, negative_rows)
    required = 2 * layers * width * itemsize * batch * cache_rows
    if torch.device(device).type != "cuda":
        return required, float("inf")

    free, _ = torch.cuda.mem_get_info(device)
    reusable = max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
    # QKV, RoPE/norm temporaries, residuals and the gated MLP at the largest
    # prefill shape. Keep an additional GiB for kernels and allocator slack.
    workspace = max(2 * GIB, (largest_prefix + target_rows) * batch * width * itemsize * 24) + GIB
    budget = max(0, free + reusable - model_storage_bytes(transformer) - workspace)
    return required, budget
