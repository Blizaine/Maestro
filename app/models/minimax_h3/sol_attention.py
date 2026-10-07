# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 policy and fail-safe wrapper for bundled Sol-Attn kernels."""

from __future__ import annotations

import torch

SOL_ATTN_TAU = 1.0
SOL_ATTN_THRESH_TYPE = "diag"
SOL_ATTN_MIN_TOKENS = 8192


class MiniMaxH3SolAttention:
    """Apply Sol only to eligible H3 DiT attention and retain dense fallback.

    The same instance is shared by all main transformer blocks. Reference,
    text, target-audio, and other prefix rows are kept as an exact KV sink.
    Short sequences, masked attention, unsupported runtimes, and any kernel
    compilation/runtime failure use Maestro's normal Sage/SDPA selection.
    """

    def __init__(self):
        self.enabled = False
        self._runtime_validated = False
        self._runtime_failed = False
        self._announced = False
        self._fallback_announced = False
        self.sink_tokens = 0

    def _fallback(self, reason: Exception | str) -> None:
        self.enabled = False
        self._runtime_failed = True
        if not self._fallback_announced:
            from shared.attention import get_default_attention_mode

            dense = get_default_attention_mode()
            print(
                "[MiniMax H3] Sol Engine could not run "
                f"({reason}); falling back to {dense}."
            )
            self._fallback_announced = True

    def begin_forward(self, sink_tokens: int, device, dtype) -> None:
        from mmgp import offload

        self.sink_tokens = max(0, int(sink_tokens))
        self.enabled = offload.shared_state.get("_attention") == "sol"
        if not self.enabled:
            return
        if self._runtime_failed:
            self.enabled = False
            return
        if self._runtime_validated:
            return
        try:
            from shared.sol_attn import validate_runtime

            capability = validate_runtime(device, dtype)
        except Exception as error:  # deterministic dense recovery
            self._fallback(error)
            return
        self._runtime_validated = True
        if not self._announced:
            print(
                "[MiniMax H3] Sol Engine enabled on "
                f"SM{capability[0]}{capability[1]} "
                f"(tau={SOL_ATTN_TAU:g}, {SOL_ATTN_THRESH_TYPE}; "
                f"exact prefix={self.sink_tokens:,} rows)."
            )
            self._announced = True

    def use_for_layer(self, tokens: int, attention_mask=None) -> bool:
        return (
            self.enabled
            and attention_mask is None
            and int(tokens) >= SOL_ATTN_MIN_TOKENS
        )

    def staged(self, x: torch.Tensor, projections) -> bool:
        """Whether this layer can use K/V-first projection and inline-Q Sol."""

        if not self.use_for_layer(x.shape[-2]):
            return False
        try:
            from mmgp import offload
            from shared.sol_attn import staged_supported

            return bool(
                staged_supported(x.device, x.shape[-2], SOL_ATTN_THRESH_TYPE)
                and offload.linear_rows_supported(projections, x)
            )
        except Exception:
            return False

    def attention(
        self,
        x_list,
        projections,
        norm_rope,
        heads: int,
        head_dim: int,
    ):
        """Compute V, then normalized K, then Q, one head group at a time."""

        from mmgp import offload
        from shared import attention_kit
        from shared.sol_attn import prepare_kv, sol_attn_prepared

        # Keep this reference until the kernel completes so a first-use Sol
        # compile failure can rebuild full Q/K/V for the dense recovery path.
        # It is the caller's tensor, not an extra activation copy.
        x = x_list.pop()
        batch, tokens, _ = x.shape
        q_proj, k_proj, v_proj = projections
        group_count = attention_kit.head_groups(heads, tokens)
        group_size = heads // group_count
        linear_input = output = project = None
        try:
            linear_input = offload.prepare_linear_input([x], projections)
            for first in range(0, heads, group_size):
                stop = first + group_size
                head_range = slice(first, stop)

                def project(module, first=first, stop=stop, prepared_input=linear_input):
                    return offload.linear_rows(
                        module,
                        prepared_input,
                        first * head_dim,
                        stop * head_dim,
                    ).view(batch, tokens, group_size, head_dim)

                value = project(v_proj)
                key = project(k_proj)
                norm_rope(None, key)
                prepared = prepare_kv(key, value)
                key = None

                query = project(q_proj)
                if stop == heads:
                    linear_input = project = None
                norm_rope(query, None)
                query = sol_attn_prepared(
                    query,
                    value,
                    prepared,
                    tau=SOL_ATTN_TAU,
                    sink_start=0,
                    sink_tokens=min(self.sink_tokens, tokens),
                    out=query,
                )
                if group_count == 1:
                    return query
                if output is None:
                    output = query.new_empty((batch, tokens, heads, head_dim))
                output[:, :, head_range].copy_(query)
                query = value = prepared = None
            return output
        except Exception as error:  # preserve the existing process-wide dense recovery
            self._fallback(error)
            linear_input = output = project = key = value = query = prepared = None
            qkv = [
                projection(x).view(batch, tokens, heads, head_dim)
                for projection in projections
            ]
            norm_rope(qkv[0], qkv[1])
            return self(qkv, False)

    def __call__(self, qkv_list, use_sol: bool):
        from shared.attention import pay_attention

        if not use_sol:
            return pay_attention(qkv_list, recycle_q=True)

        query, key, value = qkv_list
        try:
            from shared.sol_attn import sol_attn

            output = sol_attn(
                query,
                key,
                value,
                tau=SOL_ATTN_TAU,
                thresh_type=SOL_ATTN_THRESH_TYPE,
                sink_start=0,
                sink_tokens=min(self.sink_tokens, int(query.shape[1])),
                int8_qk=True,
            )
        except Exception as error:  # first-run compilation can fail safely
            self._fallback(error)
            return pay_attention(qkv_list, recycle_q=True)
        qkv_list.clear()
        return output


__all__ = [
    "MiniMaxH3SolAttention",
    "SOL_ATTN_MIN_TOKENS",
    "SOL_ATTN_TAU",
]
