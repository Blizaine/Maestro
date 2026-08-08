"""Memory-conscious First Block Cache state for MiniMax H3."""

from __future__ import annotations

import torch


class MiniMaxH3FirstBlockCache:
    """Reuse the expensive block-stack tail when the first block is stable."""

    MAX_SIGNATURE_ELEMENTS = 1 << 20

    def __init__(self, cache):
        self.cache = cache
        self.threshold = float(cache.threshold)
        self.start_step = int(cache.start_step)
        self.step = -1
        self.head_signature = None
        self.tail_residual = None
        print(
            "[MiniMax H3 Perf] First Block Cache enabled "
            f"(threshold={self.threshold:g}, start_step={self.start_step})."
        )

    def begin_step(self, step: int) -> None:
        self.step = int(step)

    def should_compute(self, signature: torch.Tensor) -> bool:
        compute = (
            self.step < self.start_step
            or self.head_signature is None
            or self.tail_residual is None
        )
        if not compute:
            previous = self.head_signature
            difference = (signature - previous).abs().mean(dtype=torch.float32)
            reference = previous.abs().mean(dtype=torch.float32).clamp_min_(1e-8)
            compute = bool((difference / reference).item() > self.threshold)
        if compute:
            self.head_signature = signature
            self.tail_residual = None
        else:
            self.cache.skipped_steps += 1
        return compute

    @staticmethod
    def capture_head_output(target_hidden: torch.Tensor) -> torch.Tensor:
        return target_hidden.clone()

    def store_tail_residual(
        self,
        target_hidden: torch.Tensor,
        head_output: torch.Tensor,
    ) -> None:
        head_output.neg_().add_(target_hidden)
        self.tail_residual = head_output

    def apply_tail_residual(self, target_hidden: torch.Tensor) -> None:
        target_hidden.add_(self.tail_residual)

    def reset(self) -> None:
        self.head_signature = None
        self.tail_residual = None
        self.step = -1


__all__ = ["MiniMaxH3FirstBlockCache"]
