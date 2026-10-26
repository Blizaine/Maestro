"""Bounded packed-expert GPU slots for one-frame Fish decoding.

Routing and misses stay outside CUDA capture. Every selected expert is resident
before the kernels run; eviction never turns an unhandled miss into zero output.
"""
from __future__ import annotations

from collections import OrderedDict

import torch


class ExpertSlotPolicy:
    """Deterministic LRU with the current route protected throughout a fill."""

    def __init__(self, capacity: int, experts: int):
        if not 1 <= capacity <= experts:
            raise ValueError("invalid Fish expert cache capacity")
        self.capacity, self.experts = capacity, experts
        self.resident = OrderedDict()

    def route(self, ids):
        selected = list(dict.fromkeys(int(index) for index in ids))
        if any(index < 0 or index >= self.experts for index in selected):
            raise ValueError("routed expert id is outside the loaded bank")
        if len(selected) > self.capacity:
            raise ValueError("selected Fish experts exceed GPU cache capacity")
        protected = set(selected)
        misses, hits = [], 0
        for expert in selected:
            if expert in self.resident:
                hits += 1
                self.resident.move_to_end(expert)
                continue
            victim = None
            if len(self.resident) < self.capacity:
                used = set(self.resident.values())
                slot = next(slot for slot in range(self.capacity) if slot not in used)
            else:
                victim = next(index for index in self.resident if index not in protected)
                slot = self.resident.pop(victim)
            self.resident[expert] = slot
            misses.append((expert, slot, victim))
        return misses, hits


def cache_slot_count(bank_bytes, expert_counts, free_bytes, total_bytes, topk=8, *, host_free_bytes=None):
    """Equal slots per bank, leaving at least 1.5 GiB after prompt/cache setup."""
    if not bank_bytes or len(bank_bytes) != len(expert_counts):
        return 0
    if any(size <= 0 or count < topk for size, count in zip(bank_bytes, expert_counts)):
        return 0
    reserve = max(1536 * 1024**2, int(total_bytes * 0.18))
    budget = max(0, int(free_bytes) - reserve)
    bytes_per_slot = sum(size // count for size, count in zip(bank_bytes, expert_counts))
    if host_free_bytes is not None and host_free_bytes < topk * bytes_per_slot + 512 * 1024**2:
        return 0
    slots = min(64, min(expert_counts), budget // max(1, bytes_per_slot))
    return int(slots) if slots >= topk else 0


class PackedExpertCache:
    """Fixed GPU allocations plus a top-k pinned staging window for one bank."""

    def __init__(self, bank, capacity, topk, device):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("expert cache requires an indexed CUDA device")
        if not bank.use_triton or len(bank.expert_tensor_names) != 4:
            raise ValueError("expert cache requires a native four-tensor Fish bank")
        self.policy = ExpertSlotPolicy(capacity, bank.num_experts)
        if capacity < topk:
            raise ValueError("Fish expert cache needs at least top-k slots")
        self.sources = tuple(getattr(bank, name) for name in bank.expert_tensor_names)
        if any(tensor.device.type != "cpu" or not tensor.is_contiguous() for tensor in self.sources):
            raise ValueError("Fish expert checkpoint tensors must remain contiguous on CPU")
        self.slots = tuple(torch.empty((capacity, *source.shape[1:]), dtype=source.dtype,
                                       device=self.device) for source in self.sources)
        self.staging = tuple(torch.empty((topk, *source.shape[1:]), dtype=source.dtype,
                                         device="cpu", pin_memory=True) for source in self.sources)
        self.host_pointers = torch.zeros((bank.num_experts, 4), dtype=torch.int64,
                                         device="cpu", pin_memory=True)
        self.pointers = torch.zeros_like(self.host_pointers, device=self.device)
        self.topk = topk
        self.hits = self.misses = self.transferred_bytes = self.frames = 0

    def prepare(self, ids):
        if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= self.topk:
            raise ValueError("Fish cached experts support one frame and bounded top-k")
        # One deliberate route read per bank. It also orders prior use of the
        # same bank's pinned staging storage before the host overwrites it.
        selected = ids.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
        misses, hits = self.policy.route(selected)
        self.hits += hits
        self.misses += len(misses)
        self.frames += 1
        if not misses:
            return self.pointers
        indices = torch.tensor([entry[0] for entry in misses], dtype=torch.long, device="cpu")
        for source, stage, slots in zip(self.sources, self.staging, self.slots):
            # Gather opaque checkpoint bytes: older CPU Torch builds do not
            # implement index_select for float8. No dequantization/conversion.
            torch.index_select(source.view(torch.uint8), 0, indices,
                               out=stage[:len(misses)].view(torch.uint8))
            for row, (expert, slot, victim) in enumerate(misses):
                slots[slot].copy_(stage[row], non_blocking=True)
                self.transferred_bytes += slots[slot].numel() * slots[slot].element_size()
        for expert, slot, victim in misses:
            if victim is not None:
                self.host_pointers[victim].zero_()
            for index, slots in enumerate(self.slots):
                self.host_pointers[expert, index] = slots[slot].data_ptr()
        self.pointers.copy_(self.host_pointers, non_blocking=True)
        return self.pointers

    def stats(self):
        return {"slots": self.policy.capacity, "hits": self.hits, "misses": self.misses,
                "routes": self.frames, "transferred_bytes": self.transferred_bytes,
                "gpu_bytes": sum(value.numel() * value.element_size() for value in self.slots)}
