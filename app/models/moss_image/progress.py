# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0
"""Small local progress bridge for MOSS inference and bounded cancellation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature


class GenerationAborted(Exception):
    """Raised at a model layer boundary when Maestro requests cancellation."""


_generation = ContextVar("moss_image_generation", default=None)


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
    state = _generation.get()
    if state is not None:
        state.check_abort()


def set_phase_status(status):
    state = _generation.get()
    if state is None:
        return
    state.check_abort()
    if callable(state.set_status):
        state.set_status(str(status))


def generation_progress(method):
    """Scope status reporting and turn a layer-boundary abort into a clean stop."""
    method_signature = signature(method)

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if _generation.get() is not None:
            return method(self, *args, **kwargs)
        arguments = method_signature.bind(self, *args, **kwargs).arguments
        state = _GenerationState(
            self,
            arguments.get("callback"),
            arguments.get("set_progress_status"),
        )
        token = _generation.set(state)
        try:
            state.check_abort()
            result = method(self, *args, **kwargs)
            return None if self._interrupt else result
        except GenerationAborted:
            return None
        finally:
            _generation.reset(token)

    return wrapped


@contextmanager
def text_encoding_progress(layers):
    """Report the text phase and poll cancellation before each decoder layer."""
    state = _generation.get()
    if state is None:
        yield
        return

    set_phase_status("Encoding Text Prompt")
    handles = []
    try:
        for layer in layers:
            handles.append(layer.register_forward_pre_hook(lambda *_: check_abort()))
        yield
        check_abort()
    finally:
        for handle in handles:
            handle.remove()
