"""Per-generation policy for the memory-saving attention chunker.

The saved mode is process-wide configuration. A generation may temporarily
override it with :func:`chunker_scope`; ContextVar keeps concurrent jobs
independent and the scope always restores its previous value.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch


CHUNKER_CONFIG_KEY = "inputs_chunker"
CHUNKER_DEFAULT = "auto"
AUTO_MARGIN_BYTES = 512 << 20
AUTO_MARGIN_SHARE = 0.10
EXTREME_MIN_SAVING = 128 << 20

_VALID_MODES = {"off", "auto", "on", "extreme"}
_saved_mode = CHUNKER_DEFAULT
_active_mode: ContextVar[str | None] = ContextVar("attention_chunker_mode", default=None)
_active_decisions: ContextVar[dict | None] = ContextVar("attention_chunker_decisions", default=None)


def normalize_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value == "standard":
        value = "on"
    if value not in _VALID_MODES:
        raise ValueError(f"Unknown Smart Inputs Chunker mode: {mode!r}")
    return value


def configure_chunker(mode: str) -> str:
    """Set the saved process default. Per-generation scopes take precedence."""

    global _saved_mode
    _saved_mode = normalize_mode(mode)
    return _saved_mode


def current_chunker() -> str:
    return _active_mode.get() or _saved_mode


@contextmanager
def chunker_scope(mode: str | None = None):
    """Set a chunker mode for one job or pass and restore it on every exit.

    A new decision cache is created at the scope boundary, so Auto decisions
    made by one generation do not affect later jobs. Nested scopes restore the
    enclosing mode and caches.
    """

    effective = current_chunker() if mode is None else normalize_mode(mode)
    mode_token = _active_mode.set(effective)
    decisions_token = _active_decisions.set({})
    try:
        yield effective
    finally:
        _active_decisions.reset(decisions_token)
        _active_mode.reset(mode_token)


def _decision_key(device, kind):
    try:
        hash(kind)
        stable_kind = kind
    except TypeError:
        stable_kind = repr(kind)
    return (device.type, device.index, stable_kind)


def auto_room(device) -> int:
    """Bytes new allocations may use after leaving room for CUDA overhead."""

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    try:
        free, card = torch.cuda.mem_get_info(device)
        allocated = int(torch.cuda.memory_allocated(device))
        reusable = max(0, int(torch.cuda.memory_reserved(device)) - allocated)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return 0
    margin = max(AUTO_MARGIN_BYTES, int(card * AUTO_MARGIN_SHARE))
    # CUDA's free-memory report accounts for the desktop and other processes.
    room = min(card - allocated, free + reusable) - margin
    try:
        from mmgp import allocator

        if allocator.active == "vmm":
            room = min(room, int(allocator.room(device)) - AUTO_MARGIN_BYTES)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return max(0, room)


def chunking_needed(device, extra_bytes: int, kind) -> bool:
    """Whether the current mode should split this call, memoized per job/shape."""

    mode = current_chunker()
    if mode == "off":
        return False
    if mode in ("on", "extreme"):
        return True
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False
    key = _decision_key(device, kind)
    decisions = _active_decisions.get()
    if decisions is not None and key in decisions:
        return decisions[key]
    needed = int(extra_bytes) > auto_room(device)
    if decisions is not None:
        decisions[key] = needed
    return needed


__all__ = [
    "AUTO_MARGIN_BYTES",
    "AUTO_MARGIN_SHARE",
    "CHUNKER_CONFIG_KEY",
    "CHUNKER_DEFAULT",
    "EXTREME_MIN_SAVING",
    "auto_room",
    "chunker_scope",
    "chunking_needed",
    "configure_chunker",
    "current_chunker",
    "normalize_mode",
]
