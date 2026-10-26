# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Fish-local compatibility for MMGP runtime features absent in older builds."""

from __future__ import annotations

from contextlib import nullcontext

import torch


_FLA_RUNTIME_CHECK_CACHE: tuple[bool, str | None] | None = None


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


def _triton_import_status() -> tuple[bool, str | None]:
    """Check Triton's Python API without querying CUDA or launching a kernel."""

    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except Exception as exc:
        return False, f"Triton import failed: {exc}"
    return True, None


def _check_fla_delta_runtime() -> tuple[bool, str | None]:
    """Run each Fish DeltaNet kernel once before enabling decode acceleration."""

    global _FLA_RUNTIME_CHECK_CACHE
    if _FLA_RUNTIME_CHECK_CACHE is not None:
        return _FLA_RUNTIME_CHECK_CACHE
    try:
        from shared.llm_engines.nanovllm.models import qwen3_5

        chunk_rule = qwen3_5._DEFAULT_FAST_CHUNK_GATED_DELTA_RULE
        recurrent_rule = qwen3_5._DEFAULT_FAST_RECURRENT_GATED_DELTA_RULE
        if not callable(chunk_rule) or not callable(recurrent_rule):
            _FLA_RUNTIME_CHECK_CACHE = (False, "FLA DeltaNet kernels are unavailable")
            return _FLA_RUNTIME_CHECK_CACHE
        if not torch.cuda.is_available():
            _FLA_RUNTIME_CHECK_CACHE = (False, "CUDA is not available for the FLA DeltaNet smoke test")
            return _FLA_RUNTIME_CHECK_CACHE

        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        # Match the model's q/k/v, beta and fp32 log-decay dtypes. Sequence
        # length 64 exercises one complete FLA chunk without materializing a
        # model-sized activation or cache.
        with torch.inference_mode():
            q = torch.zeros((1, 64, 1, 64), device=device, dtype=dtype)
            k = torch.zeros_like(q)
            v = torch.zeros_like(q)
            decay = torch.full((1, 64, 1), -0.01, device=device, dtype=torch.float32)
            beta = torch.full((1, 64, 1), 0.5, device=device, dtype=dtype)
            chunk_output, chunk_state = chunk_rule(
                q,
                k,
                v,
                g=decay,
                beta=beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            recurrent_output, recurrent_state = recurrent_rule(
                q[:, :1],
                k[:, :1],
                v[:, :1],
                g=decay[:, :1],
                beta=beta[:, :1],
                initial_state=torch.zeros((1, 1, 64, 64), device=device, dtype=torch.float32),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            torch.cuda.synchronize(device)
            results = (chunk_output, chunk_state, recurrent_output, recurrent_state)
            if any(not bool(torch.isfinite(value).all().item()) for value in results):
                _FLA_RUNTIME_CHECK_CACHE = (False, "FLA DeltaNet smoke test returned non-finite output")
                return _FLA_RUNTIME_CHECK_CACHE
    except Exception as exc:
        message = str(exc).replace("\n", " ").strip()
        if len(message) > 260:
            message = message[:260] + "..."
        _FLA_RUNTIME_CHECK_CACHE = (False, f"FLA DeltaNet smoke test failed: {message}")
        return _FLA_RUNTIME_CHECK_CACHE
    _FLA_RUNTIME_CHECK_CACHE = (True, None)
    return _FLA_RUNTIME_CHECK_CACHE


def fish_vllm_availability(*, verify_runtime: bool = False) -> tuple[bool, str | None]:
    """Check Fish's SDPA, Triton and FLA requirements.

    Model discovery uses structural checks only. The optional Triton runtime
    smoke runs when the user actually selects this engine, before model
    loading, expert-cache setup, or CUDA graph capture.
    """

    triton_ok, triton_reason = _triton_import_status()
    if not triton_ok:
        return False, f"Triton runtime: {triton_reason or 'unavailable'}"

    if not callable(getattr(torch.nn.functional, "scaled_dot_product_attention", None)):
        return False, "PyTorch SDPA: scaled_dot_product_attention is unavailable"

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
        if not callable(function)
    ]
    if missing:
        return False, "Fish requires FLA kernels: " + ", ".join(missing)
    if verify_runtime:
        try:
            runtime_ok, runtime_reason = _check_fla_delta_runtime()
        except Exception as exc:
            return False, f"FLA DeltaNet runtime smoke failed: {exc}"
        if not runtime_ok:
            return False, f"FLA DeltaNet runtime: {runtime_reason or 'smoke test failed'}"
    return True, None


def can_place_gpu_experts(required_bytes: int, free_bytes: int, total_bytes: int, profile: int) -> bool:
    """Keep a cache/activation reserve and only stage experts for full-VRAM profiles."""

    if int(profile) not in (1, 3):
        return False
    reserve = max(4 * 1024**3, int(total_bytes * 0.15))
    return int(free_bytes) >= int(required_bytes) + reserve
