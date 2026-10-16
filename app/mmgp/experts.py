# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Marker types for routed expert tensors kept outside ordinary MMGP blocks.

Fish Audio owns its expert execution and any optional accelerated copies.
This module does not create CUDA copies or install hooks: the MMGP offload
manager uses :func:`excluded_names` to leave the independently managed banks
out of its normal block/device handling.
"""

from __future__ import annotations

from torch import nn


class ExpertBank(nn.Module):
    """Base class for checkpoint tensors whose leading axis indexes experts."""

    def __init__(self, num_experts: int, tensor_names=()) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.expert_tensor_names = tuple(tensor_names)
        self._fish_cpu_expert_bank = True

    @property
    def expert_nbytes(self) -> int:
        return sum(
            getattr(self, name).numel() * getattr(self, name).element_size()
            for name in self.expert_tensor_names
        )

    def _apply(self, fn, recurse=True):
        """Keep checkpoint storage on CPU when MMGP moves surrounding modules.

        MMGP's Fish integration excludes banks from its per-block movement too;
        this guard also protects against a later whole-model ``to('cuda')``.
        State-dict assignment still installs the CPU mmap tensors directly.
        """

        return self


def banks(model: nn.Module) -> dict[str, ExpertBank]:
    """Return expert banks registered below ``model`` by qualified name."""

    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, ExpertBank)
    }


def excluded_names(model: nn.Module) -> set[str]:
    """Return each expert bank and descendant, for MMGP's block-hook loop."""

    prefixes = tuple(banks(model))
    if not prefixes:
        return set()
    return {
        name
        for name, _ in model.named_modules()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    }
