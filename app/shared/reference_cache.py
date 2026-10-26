"""Bounded reference K/V storage for diffusion models.

The optional symmetric INT8 layout follows WanGP's per-token, per-head,
32-channel prefix-cache approach (9905f0df941e5ec7b0c80e32555d3a61fe415726,
shared/attention_kit.py). This portable path uses FP32 scales and the existing
attention backend; it does not require the upstream staged Sage kernel API.
WanGP additions retain its Community License 2.0; see reference_cache.NOTICE.
"""
from __future__ import annotations

import torch

CACHE_MODES = ("Disabled", "Enabled", "INT8", "INT8_RAM")
GROUP_SIZE = 32
CHUNK_ROWS = 32768


def normalize_cache_mode(mode):
    value = "Disabled" if mode is None else str(mode)
    if value not in CACHE_MODES:
        raise ValueError(f"Unknown reference KV cache mode: {value!r}")
    return value


def quantized_storage_bytes(shape):
    """Include every scale and partial channel group, not just the INT8 data."""
    rows = 1
    for dimension in shape[:-1]:
        rows *= dimension
    channels = shape[-1]
    return rows * (channels + 4 * ((channels + GROUP_SIZE - 1) // GROUP_SIZE))


class QuantizedReferenceTensor:
    """One [batch, tokens, heads, channels] prefix, with bounded FP32 scratch."""

    def __init__(self, tensor, *, ram=False):
        if tensor.ndim != 4 or not tensor.is_floating_point():
            raise ValueError("Reference KV cache requires a four-dimensional floating tensor")
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.device = tensor.device
        channels = self.shape[-1]
        groups = (channels + GROUP_SIZE - 1) // GROUP_SIZE
        storage_device = torch.device("cpu") if ram else tensor.device
        self.data = torch.empty(self.shape, dtype=torch.int8, device=storage_device)
        self.scales = torch.empty((*self.shape[:-1], groups), dtype=torch.float32, device=storage_device)
        if channels == 0 or self.shape[2] == 0:
            raise ValueError("Reference KV cache requires nonempty heads and channels")
        token_chunk = max(1, CHUNK_ROWS // self.shape[2])
        finite = torch.ones((), dtype=torch.bool, device=tensor.device)
        for batch in range(self.shape[0]):
            for start in range(0, self.shape[1], token_chunk):
                stop = min(start + token_chunk, self.shape[1])
                # Slice before reshape so even noncontiguous inputs require
                # only a bounded copy, never a full-prefix FP32 temporary.
                block = tensor[batch, start:stop].reshape(-1, channels).float()
                finite.logical_and_(torch.isfinite(block).all())
                if channels % GROUP_SIZE:
                    block = torch.nn.functional.pad(block, (0, groups * GROUP_SIZE - channels))
                grouped = block.view(-1, groups, GROUP_SIZE)
                scale = grouped.abs().amax(-1).div(127).clamp_min(torch.finfo(torch.float32).tiny)
                values = grouped.div(scale.unsqueeze(-1)).round().clamp(-127, 127)
                # RAM copies are synchronous. Do not impose a pinned-memory
                # policy or asynchronous source lifetime on the caller.
                self.data[batch, start:stop].copy_(
                    values.flatten(1)[:, :channels].to(torch.int8).view(stop - start, self.shape[2], channels))
                self.scales[batch, start:stop].copy_(scale.view(stop - start, self.shape[2], groups))
        # One host synchronization per tensor, rather than per chunk.
        if not bool(finite):
            raise ValueError("Non-finite reference activations cannot be cached as INT8")

    @property
    def storage_bytes(self):
        return self.data.numel() + self.scales.numel() * self.scales.element_size()

    def materialize(self, *, device=None, dtype=None):
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        result = torch.empty(self.shape, dtype=dtype, device=device)
        channels = self.shape[-1]
        groups = self.scales.shape[-1]
        for batch in range(self.shape[0]):
            data = self.data[batch].view(-1, channels)
            scales = self.scales[batch].view(-1, groups)
            output = result[batch].view(-1, channels)
            for start in range(0, data.shape[0], CHUNK_ROWS):
                stop = min(start + CHUNK_ROWS, data.shape[0])
                values = data[start:stop].to(device=device, dtype=torch.float32)
                scale = scales[start:stop].to(device=device)
                values.mul_(scale.repeat_interleave(GROUP_SIZE, dim=-1)[:, :channels])
                output[start:stop].copy_(values)
        return result


class ReferenceKVLayerCache:
    def __init__(self, storage_mode="Enabled"):
        self.storage_mode = normalize_cache_mode(storage_mode)
        self.k = self.v = None

    def store(self, key, value):
        if self.storage_mode == "Disabled":
            raise ValueError("A disabled reference cache cannot store tensors")
        if self.storage_mode == "Enabled":
            # A prefix view can otherwise retain every target token too.
            self.k, self.v = key.clone(), value.clone()
        else:
            ram = self.storage_mode == "INT8_RAM"
            self.k = QuantizedReferenceTensor(key, ram=ram)
            self.v = QuantizedReferenceTensor(value, ram=ram)

    def get(self, *, device=None, dtype=None):
        if self.k is None or self.v is None:
            raise RuntimeError("KV cache has not been populated yet")
        if self.storage_mode == "Enabled":
            return self.k.to(device=device, dtype=dtype), self.v.to(device=device, dtype=dtype)
        return self.k.materialize(device=device, dtype=dtype), self.v.materialize(device=device, dtype=dtype)

    def clear(self):
        self.k = self.v = None
