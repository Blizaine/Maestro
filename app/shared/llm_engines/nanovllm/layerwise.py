# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Attention and DeltaNet helpers used by hybrid autoregressive decoders.

The ordinary path uses PyTorch SDPA and the Fish model's reference DeltaNet
implementation.  FlashAttention/FLA are imported only for an explicitly
selected accelerated engine.
"""

from __future__ import annotations

import torch

from shared.attention import pay_attention


def cached_attention(
    qkv,
    group,
    cache,
    positions,
    kv_len: int,
    engine: str,
    *,
    attention_backend: str | None = None,
):
    query, key, value = qkv
    qkv.clear()
    key_cache, value_cache = cache
    start, stop = group.kv.start, group.kv.stop
    heads = stop - start
    if heads <= 0 or query.shape[2] % heads:
        raise ValueError("attention query heads must be divisible by the key/value head group")
    groups = query.shape[2] // heads
    # Maestro's attention kit passes native KV heads; newer WanGP kits
    # expand them to query heads before the callback. Cache each KV head once.
    if key.shape[2] == query.shape[2] and groups > 1:
        key, value = key[:, :, ::groups], value[:, :, ::groups]
    if key.shape[2] != heads or value.shape[2] != heads:
        raise ValueError("attention cache received the wrong number of key/value heads")
    key_cache[:, :, start:stop].index_copy_(1, positions, key)
    value_cache[:, :, start:stop].index_copy_(1, positions, value)
    if attention_backend not in (None, "sdpa"):
        raise ValueError(f"unsupported layerwise attention backend: {attention_backend}")
    if engine == "vllm" and attention_backend != "sdpa":
        from .layers.attention import _DEFAULT_FLASH_ATTN_WITH_KVCACHE

        if _DEFAULT_FLASH_ATTN_WITH_KVCACHE is None:
            raise RuntimeError("the vLLM layerwise engine requires FlashAttention 2")
        lengths = (positions[-1:] + 1).to(dtype=torch.int32).expand(query.shape[0]).contiguous()
        return _DEFAULT_FLASH_ATTN_WITH_KVCACHE(
            query,
            key_cache[:, :, start:stop],
            value_cache[:, :, start:stop],
            cache_seqlens=lengths,
            causal=True,
            num_splits=1 if query.shape[1] > 1 else 0,
        )

    key = key_cache[:, :kv_len, start:stop]
    value = value_cache[:, :kv_len, start:stop]
    if groups > 1:
        key = key.repeat_interleave(groups, dim=2)
        value = value.repeat_interleave(groups, dim=2)
    mask = torch.arange(kv_len, device=query.device)[None, :] <= positions[:, None]
    return pay_attention(
        [query, key, value],
        attention_mask=mask[None, :, None, :],
        force_attention="sdpa",
    )


def delta_rule(query, key, value, decay, beta, initial_state, *, engine, recurrent, torch_impl):
    if engine == "vllm":
        from .models import qwen3_5

        # Older Maestro FLA integrations already expose the inference kernels
        # without WanGP's optional autotuner customization.
        configure = getattr(qwen3_5, "configure_qwen35_fla_prefill_autotune", None)
        if callable(configure):
            configure(query.device)
        function = (
            qwen3_5._DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE
            if recurrent
            else qwen3_5._DEFAULT_FAST_CHUNK_GATED_DELTA_RULE
        )
        if function is None:
            raise RuntimeError("Fish Audio's accelerated engine requires the shared FLA DeltaNet kernels")
        return function(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g=decay.contiguous(),
            beta=beta.contiguous(),
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    return torch_impl(query, key, value, decay, beta, initial_state)
