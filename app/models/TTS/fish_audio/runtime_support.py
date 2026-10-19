# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Fish-local compatibility for MMGP runtime features absent in older builds."""

from __future__ import annotations

from contextlib import nullcontext

import torch


try:
    from mmgp import runtime as runtime
except ImportError:  # pragma: no cover - normally provided by MMGP integration
    class _RuntimeFallback:
        @staticmethod
        def execution_device(module):
            forced = getattr(module, "_force_device", None)
            if forced is not None:
                return torch.device(forced)
            for tensor in (*module.parameters(), *module.buffers()):
                if not tensor.is_meta:
                    return tensor.device
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        @staticmethod
        def execution_scope(check_abort=None):
            return nullcontext()

        @staticmethod
        def ensure_loaded(module):
            return None

        @staticmethod
        def weights_resident(module):
            return False

        @staticmethod
        def place_experts(module):
            return False

    runtime = _RuntimeFallback()


try:
    from mmgp import cuda_graphs as cuda_graphs
except ImportError:  # pragma: no cover - portable legacy fallback
    class _CudaGraphsFallback:
        @staticmethod
        def scope(enabled=True):
            return nullcontext()

        @staticmethod
        def run(module, *args, method=None, enabled=True, **kwargs):
            function = getattr(module, method) if method else module
            return function(*args, **kwargs)

        @staticmethod
        def is_enabled(module=None):
            return False

        @staticmethod
        def invalidate(*args, **kwargs):
            return None

    cuda_graphs = _CudaGraphsFallback()


def fish_vllm_availability() -> tuple[bool, str | None]:
    """Require the actual attention and DeltaNet kernels used by Fish's fast path."""

    try:
        from shared.llm_engines.nanovllm.vllm_support import probe_vllm_runtime

        probe = probe_vllm_runtime()
    except Exception as exc:
        return False, f"shared accelerated-engine probe failed: {exc}"
    if not probe.get("supported", False):
        checks = probe.get("checks", {})
        failures = [
            f"{name}: {result.get('message', 'unavailable')}"
            for name, result in checks.items()
            if isinstance(result, dict) and not result.get("ok", False)
        ]
        return False, "; ".join(failures) or "Triton and FlashAttention 2 are required"

    try:
        from shared.llm_engines.nanovllm.layers import attention as shared_attention
    except Exception as exc:
        return False, f"shared FlashAttention layer failed to import: {exc}"
    missing_attention = [
        name
        for name, function in (
            ("flash_attn_varlen_func", shared_attention._DEFAULT_FLASH_ATTN_VARLEN_FUNC),
            ("flash_attn_with_kvcache", shared_attention._DEFAULT_FLASH_ATTN_WITH_KVCACHE),
        )
        if not callable(function)
    ]
    if missing_attention:
        return False, "Fish requires FlashAttention 2 functions: " + ", ".join(missing_attention)

    try:
        from shared.llm_engines.nanovllm.models import qwen3_5
    except Exception as exc:
        return False, f"shared FLA DeltaNet kernels failed to import: {exc}"
    missing = [
        name
        for name, function in (
            ("chunk_gated_delta_rule", qwen3_5._DEFAULT_FAST_CHUNK_GATED_DELTA_RULE),
            ("fused_recurrent_gated_delta_rule", qwen3_5._DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE),
        )
        if function is None
    ]
    if missing:
        return False, "Fish requires FLA kernels: " + ", ".join(missing)
    return True, None


def can_place_gpu_experts(required_bytes: int, free_bytes: int, total_bytes: int, profile: int) -> bool:
    """Keep a cache/activation reserve and only stage experts for full-VRAM profiles."""

    if int(profile) not in (1, 3):
        return False
    reserve = max(4 * 1024**3, int(total_bytes * 0.15))
    return int(free_bytes) >= int(required_bytes) + reserve
