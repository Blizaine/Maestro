# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Coordinate reclaimable VRAM: optional expert copies precede streamed blocks."""
import weakref

from . import allocator

_clients = weakref.WeakKeyDictionary()
HEADROOM = 256 << 20  # the allocator leaves this much free beyond a new allocation (vram_headroom)


def _reclaim(size, device):
    # what the allocation of size bytes misses, the VRAM it may still take deducted (its size when the room did not refuse it)
    room = allocator.room(device)
    size = size + HEADROOM - room if room < size + HEADROOM else size
    freed = 0
    for client, _ in sorted(list(_clients.items()), key=lambda item: item[1]):
        freed += client.on_pressure(max(0, size - freed))
        if freed >= size:
            break
    return freed


def register(client, priority=1):
    _clients[client] = priority
    if allocator.active == "vmm":
        allocator.set_pressure_callback(_reclaim)


def unregister(client):
    _clients.pop(client, None)
    if not _clients and allocator.active == "vmm":
        allocator.set_pressure_callback(None)
