# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Local progress and cancellation hooks for Hy-Nomas layers."""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from inspect import signature


class GenerationAborted(Exception):
    """Raised at a model layer boundary when Maestro requests cancellation."""


_active = ContextVar("hy_nomas_generation", default=None)


class _GenerationState:
    def __init__(self, pipeline, callback, set_status):
        self.pipeline = pipeline
        self.callback = callback
        self.set_status = set_status

    def check_abort(self):
        checkpoint = getattr(self.callback, "checkpoint", None)
        if callable(checkpoint):
            checkpoint()
        if self.pipeline._interrupt:
            raise GenerationAborted


def check_abort():
    state = _active.get()
    if state is not None:
        state.check_abort()


def set_phase_status(status):
    state = _active.get()
    if state is None:
        return
    state.check_abort()
    if callable(state.set_status):
        state.set_status(str(status))


def generation_progress(method):
    method_signature = signature(method)

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if _active.get() is not None:
            return method(self, *args, **kwargs)
        arguments = method_signature.bind(self, *args, **kwargs).arguments
        options = arguments.get("kwargs") or {}
        state = _GenerationState(
            self,
            arguments.get("callback") or options.get("callback"),
            arguments.get("set_progress_status") or options.get("set_progress_status"),
        )
        token = _active.set(state)
        try:
            state.check_abort()
            result = method(self, *args, **kwargs)
            return None if self._interrupt else result
        except GenerationAborted:
            return None
        finally:
            _active.reset(token)

    return wrapped


class vae_decoding_progress:
    """Poll cancellation at each decoder window's final projection."""

    def __init__(self, total, anchor):
        self.total = max(0, int(total))
        self.anchor = anchor
        self.handle = None

    def __enter__(self):
        set_phase_status("Decoding World Video")
        state = _active.get()
        if state is not None and hasattr(self.anchor, "register_forward_pre_hook"):
            self.handle = self.anchor.register_forward_pre_hook(lambda *_: check_abort())
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.remove()
        check_abort()
        return False
