# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Small Fish-local bridge to Maestro's generation status callback."""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps


_status_callback = ContextVar("fish_audio_status_callback", default=None)


def set_phase_status(status: str) -> None:
    callback = _status_callback.get()
    if callable(callback):
        text = str(status or "").strip()
        if text:
            callback(text)


def generation_progress(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _status_callback.set(kwargs.get("set_progress_status"))
        try:
            return function(*args, **kwargs)
        finally:
            _status_callback.reset(token)

    return wrapped


class PhaseProgress:
    """Context-compatible status reporter; frame-level callbacks stay with the caller."""

    def __init__(self, total, label, unit, next_status=None):
        self.total = max(0, int(total))
        self.label = str(label)
        self.unit = str(unit)
        self.next_status = next_status
        self.completed = 0

    def __enter__(self):
        set_phase_status(self.label)
        return self

    def advance(self, amount=1):
        self.completed = min(self.total, self.completed + int(amount))

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None and self.next_status:
            set_phase_status(self.next_status)
        return False
