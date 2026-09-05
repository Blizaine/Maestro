# SPDX-License-Identifier: MIT
# Copyright (c) 2026 PlagueKind
"""Runtime policy and fail-safe wrapper for MiniMax H3 SLA attention."""

from __future__ import annotations

from typing import Any


DEFAULT_SLA_CONFIG = {
    "sparsity_ratio": 0.90,
    "block_size": 64,
    "min_seq_len": 8192,
    "dense_last_steps": 0,
    "protect_audio": True,
}


def normalize_sla_config(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    result = dict(DEFAULT_SLA_CONFIG)
    result.update({key: raw[key] for key in result if key in raw})
    result["sparsity_ratio"] = min(
        0.95,
        max(0.0, float(result["sparsity_ratio"])),
    )
    result["block_size"] = int(result["block_size"])
    if result["block_size"] not in {64, 128}:
        result["block_size"] = 64
    result["min_seq_len"] = max(1, int(result["min_seq_len"]))
    result["dense_last_steps"] = max(0, int(result["dense_last_steps"]))
    result["protect_audio"] = bool(result["protect_audio"])
    return result


class MiniMaxH3SLAAttention:
    """Apply SLA only to eligible H3 DiT attention with dense recovery."""

    def __init__(self, config: Any = None):
        self.config = normalize_sla_config(config)
        self.enabled = False
        self.sink_tokens = 0
        self.step_index = 0
        self.total_steps = 1
        self._runtime_failed = False
        self._runtime_validated = False
        self._announced = False
        self._fallback_announced = False
        self._calls = 0
        self._dense_calls = 0
        self._last_topk = 0
        self._last_blocks = 0

    def _fallback(self, reason: Exception | str) -> None:
        self.enabled = False
        self._runtime_failed = True
        if not self._fallback_announced:
            from shared.attention import get_default_attention_mode

            print(
                "[MiniMax H3 SLA] Sparse attention could not run "
                f"({reason}); falling back to {get_default_attention_mode()}."
            )
            self._fallback_announced = True

    def begin_step(self, step_index: int, total_steps: int) -> None:
        self.step_index = max(0, int(step_index))
        self.total_steps = max(1, int(total_steps))

    def begin_forward(self, sink_tokens: int, device, dtype) -> None:
        from mmgp import offload

        requested = offload.shared_state.get("_attention") == "sla"
        self.enabled = bool(requested and not self._runtime_failed)
        self.sink_tokens = max(0, int(sink_tokens))
        if not self.enabled or self._runtime_validated:
            return
        try:
            from shared.attention import get_sla_attention_status

            status = get_sla_attention_status()
            if not status.get("supported"):
                raise RuntimeError(status.get("reason") or "unsupported runtime")
            if str(getattr(device, "type", device)) != "cuda":
                raise RuntimeError("SLA requires CUDA tensors")
            import torch

            if dtype not in {torch.float16, torch.bfloat16}:
                raise RuntimeError(f"SLA requires FP16/BF16 attention, got {dtype}")
        except Exception as error:
            self._fallback(error)
            return
        self._runtime_validated = True
        if not self._announced:
            config = self.config
            print(
                "[MiniMax H3 SLA] Enabled "
                f"({config['sparsity_ratio'] * 100:.0f}% requested sparsity, "
                f"block {config['block_size']}, exact prefix "
                f"{self.sink_tokens:,} rows). First use compiles Triton kernels."
            )
            self._announced = True

    def use_for_layer(self, tokens: int, attention_mask=None) -> bool:
        dense_last = int(self.config["dense_last_steps"])
        trailing_dense = dense_last and self.step_index >= self.total_steps - dense_last
        eligible = (
            self.enabled
            and attention_mask is None
            and int(tokens) >= int(self.config["min_seq_len"])
            and not trailing_dense
        )
        if not eligible:
            self._dense_calls += 1
        return bool(eligible)

    def __call__(self, qkv_list, use_sla: bool):
        from shared.attention import pay_attention

        if not use_sla:
            return pay_attention(qkv_list, recycle_q=True)
        query, key, value = qkv_list
        try:
            from .sla_block_map import get_block_map
            from .sla_kernel import block_sparse_attention

            if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous()):
                query, key, value = (
                    tensor.contiguous() for tensor in (query, key, value)
                )
            block_query = int(self.config["block_size"])
            block_key = 64 if block_query == 128 else block_query
            prefix = (
                min(self.sink_tokens, int(query.shape[1]) - 1)
                if self.config["protect_audio"] and query.shape[1] > 1
                else 0
            )
            lookup, topk = get_block_map(
                query,
                key,
                1.0 - float(self.config["sparsity_ratio"]),
                block_query,
                block_key,
                protect_upto=prefix,
            )
            output = block_sparse_attention(
                query,
                key,
                value,
                lookup,
                topk,
                block_query,
                block_key,
            )
            self._calls += 1
            self._last_topk = int(topk)
            self._last_blocks = (
                int(query.shape[1]) + block_key - 1
            ) // block_key
        except Exception as error:  # compilation and runtime failures are safe
            self._fallback(error)
            return pay_attention(qkv_list, recycle_q=True)
        qkv_list.clear()
        return output

    def summary(self) -> str:
        if not self._calls or not self._last_blocks:
            return "dense fallback"
        sparse = 1.0 - self._last_topk / self._last_blocks
        return (
            f"{self._calls} sparse calls, {sparse * 100:.1f}% effective "
            f"block sparsity, {self._dense_calls} dense fall-throughs"
        )


__all__ = [
    "DEFAULT_SLA_CONFIG",
    "MiniMaxH3SLAAttention",
    "normalize_sla_config",
]
