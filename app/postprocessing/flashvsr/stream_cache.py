# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt).
# The streaming-cache design is adapted from WanGP v19.25's FlashVSR stream cache;
# this implementation uses the pinned MMGP v43 StateRing store/fetch API.
"""Bounded per-block KV state for FlashVSR streaming chunks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


def _state_ring_for_modules(modules: Sequence[torch.nn.Module], name: str):
    """Create a v43 state ring for a subset of a managed model's blocks.

    MMGP v43's public ``runtime.state_ring`` accepts a registered ModuleList,
    whereas FlashVSR+ groups blocks by cache window. Build the same ring with
    the exact manager entries for each selected block so its fetches still
    follow the model's weight-prefetch schedule.
    """
    from mmgp import runtime
    from mmgp.state_ring import StateRing

    owner_for = getattr(runtime, "_owner", None)
    owner = owner_for(modules[0]) if callable(owner_for) else None
    device = torch.device("cuda", torch.cuda.current_device())
    if owner is None:
        return StateRing(len(modules), device, None, name, modules=modules)

    manager, model_id = owner
    model = manager.models[model_id]
    module_names = {id(module): module_name for module_name, module in model.named_modules()}
    try:
        entries = [f"{model_id}/{module_names[id(module)]}" for module in modules]
    except KeyError as error:
        raise RuntimeError("FlashVSR cache blocks are missing from their MMGP-managed model") from error
    return StateRing(
        len(modules), device, manager, name, modules=modules,
        model_id=model_id, entries=entries, resident=False,
    )


class StreamCache:
    """Keep each block's last attention windows in MMGP's prefetchable state ring.

    The cache capacity is fixed per block, but its logical length can grow on
    the first FlashVSR+ Turbo chunk (two windows) to the requested length on
    later chunks. MMGP v43 only exposes ``store``/``fetch``, so an update
    stores the current bounded state again; block hooks prefetch it with the
    associated weights on the next pass.
    """

    def __init__(
        self,
        blocks: Sequence[torch.nn.Module],
        windows: Sequence[int],
        window_tokens: int,
        heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        ring_factory: Callable[[Sequence[torch.nn.Module], str], Any] | None = None,
    ) -> None:
        if len(blocks) != len(windows) or not blocks:
            raise ValueError("FlashVSR cache windows must match its non-empty block list")
        if min(int(window_tokens), int(heads), int(head_dim)) <= 0:
            raise ValueError("FlashVSR cache dimensions must be positive")
        self.windows = [max(1, int(count)) for count in windows]
        self.window_tokens = int(window_tokens)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.device = torch.device(device)
        self.storage_device = torch.device("cpu")
        self.rings: dict[int, Any] = {}
        self.positions: dict[int, tuple[Any, int]] = {}
        self.handles: dict[int, list[torch.Tensor]] = {}
        self.cpu_handles: dict[int, list[torch.Tensor]] = {}
        self.lengths: dict[int, int] = {}

        if ring_factory is None and self.device.type == "cuda" and torch.cuda.is_available():
            try:
                from mmgp import runtime as mmgp_runtime
                if callable(getattr(mmgp_runtime, "state_ring", None)):
                    ring_factory = _state_ring_for_modules
            except ImportError:
                # Legacy MMGP 4.0 has no state-ring API. Keep the previous
                # bounded CPU cache and copy only the active block to CUDA.
                pass
        if ring_factory is not None:
            try:
                for kept in sorted(set(self.windows)):
                    members = [index for index, count in enumerate(self.windows) if count == kept]
                    ring = ring_factory(
                        [blocks[index] for index in members],
                        f"FlashVSR stream cache ({kept} window{'s' if kept != 1 else ''})",
                    )
                    self.rings[kept] = ring
                    for ring_index, block in enumerate(members):
                        self.positions[block] = ring, ring_index
            except Exception as error:
                for ring in self.rings.values():
                    ring.release()
                self.rings.clear()
                self.positions.clear()
                print(f"[FlashVSR] MMGP state-ring cache unavailable ({error}); using the bounded host cache")
            else:
                self.storage_device = self.device

    @property
    def storing(self) -> bool:
        return not self.lengths

    def prefix(
        self,
        block: int,
        block_tokens: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        rows = self.lengths.get(block)
        if rows is None:
            return None
        if rows % block_tokens:
            raise RuntimeError("FlashVSR cache rows must contain whole attention windows")
        state = self.handles.get(block) or self.cpu_handles[block]
        return tuple(
            tensor[:rows].reshape(-1, block_tokens, dim).to(device=device, dtype=dtype)
            for tensor in state
        )

    def commit(self, block: int, key: torch.Tensor, value: torch.Tensor, block_tokens: int) -> None:
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("FlashVSR stream cache expects matching window tensors")
        if key.shape[1] != block_tokens or key.shape[2] != self.heads * self.head_dim:
            raise ValueError("FlashVSR stream cache tensor geometry does not match its configuration")

        keys = key.reshape(-1, self.heads, self.head_dim)
        values = value.reshape(-1, self.heads, self.head_dim)
        rows = int(keys.shape[0])
        capacity = self.windows[block] * self.window_tokens
        if rows > capacity:
            raise ValueError(f"FlashVSR cache block {block} exceeded its configured window capacity")

        handles = self.handles.get(block) or self.cpu_handles.get(block)
        if handles is None:
            handles = [
                torch.zeros((capacity, self.heads, self.head_dim), dtype=self.dtype, device=self.storage_device),
                torch.zeros((capacity, self.heads, self.head_dim), dtype=self.dtype, device=self.storage_device),
            ]
        handles[0][:rows].copy_(keys.to(device=handles[0].device, dtype=handles[0].dtype))
        handles[1][:rows].copy_(values.to(device=handles[1].device, dtype=handles[1].dtype))
        ring_position = self.positions.get(block)
        if ring_position is not None:
            ring, ring_index = ring_position
            saved = ring.store(ring_index, handles, current=True)
            self.handles[block] = saved
        else:
            self.cpu_handles[block] = handles
        self.lengths[block] = rows

    def release(self) -> None:
        for ring in self.rings.values():
            ring.release()
        self.rings.clear()
        self.positions.clear()
        self.handles.clear()
        self.cpu_handles.clear()
        self.lengths.clear()


__all__ = ["StreamCache"]
