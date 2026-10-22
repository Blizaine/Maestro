"""Local, request-scoped writer drafts and actionable rejection reasons.

The job/project owns persistence. This module never writes prompts to a shared
log or sends them elsewhere, and diagnostics never control generation limits.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import time
import json
import os
from pathlib import Path
import threading


_disk_lock = threading.RLock()


class DraftHistoryArchive:
    """Atomic local history for interactive requests which have no queue job."""
    def __init__(self, directory):
        self.directory = Path(directory)

    def _path(self, history_id):
        if not isinstance(history_id, str) or not history_id or not all(
                character in "0123456789abcdef-" for character in history_id):
            raise ValueError("Invalid writer history ID")
        return self.directory / f"{history_id}.json"

    def save(self, history_id, record):
        with _disk_lock:
            target = self._path(history_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)

    def load(self, history_id):
        with _disk_lock:
            try:
                return json.loads(self._path(history_id).read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None


_scope = ContextVar("enhancement_diagnostics", default=None)
_active = ContextVar("enhancement_diagnostic_call", default=None)


def _changed():
    scope = _scope.get()
    if scope and scope["save"]:
        try:
            scope["save"](deepcopy(scope["drafts"]))
        except Exception as error:
            # A diagnostic storage failure must not discard a usable plan.
            print(f"[LLM diagnostics] Could not checkpoint writer drafts: {error}")


@contextmanager
def capture_drafts(drafts=None, on_change=None):
    """Capture calls in this context, including async-to-thread propagation."""
    existing = _scope.get()
    if existing is not None:
        yield existing["drafts"]
        return
    scope = {"drafts": deepcopy(drafts or []), "save": on_change}
    token = _scope.set(scope)
    try:
        yield scope["drafts"]
    finally:
        _changed()
        _scope.reset(token)


def record_payload(payload, *, model_id=None):
    entry = _active.get()
    if entry is not None:
        entry["model_id"] = model_id
        # Do not duplicate image data, credentials or transport configuration.
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if isinstance(messages, list):
            entry["input_messages"] = []
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(part.get("text", "") for part in content
                                        if isinstance(part, dict) and part.get("type") == "text")
                entry["input_messages"].append({"role": message.get("role"), "content": content})


def record_response(content, reasoning="", *, complete=True):
    entry = _active.get()
    if entry is not None:
        entry.update(response_text=str(content or ""), reasoning_text=str(reasoning or ""),
                     response_complete=bool(complete))
        _changed()


def record_metrics(metrics):
    entry = _active.get()
    if entry is not None and isinstance(metrics, dict):
        entry["metrics"] = {key: metrics[key] for key in (
            "prompt_tokens", "completion_tokens", "reasoning_tokens", "answer_tokens",
            "finish_reason", "max_new_tokens",
        ) if key in metrics}


def record_rejection(reasons, *, stage=""):
    """Attach a validation decision to the most recent writer response."""
    scope = _scope.get()
    if not scope or not scope["drafts"]:
        return
    entry = scope["drafts"][-1]
    reasons = [str(reason) for reason in reasons if str(reason).strip()]
    if not reasons:
        return
    entry["status"] = "rejected"
    entry.setdefault("rejections", []).append({"stage": stage, "reasons": reasons})
    _changed()


def diagnostic_call(function):
    """Record each transport call once, including delegated streaming calls."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        scope = _scope.get()
        if scope is None or _active.get() is not None:
            return function(*args, **kwargs)
        entry = {"index": len(scope["drafts"]) + 1, "function": function.__name__,
                 "status": "running", "response_text": "", "rejections": []}
        scope["drafts"].append(entry)
        token = _active.set(entry)
        started = time.monotonic()
        try:
            result = function(*args, **kwargs)
            entry.setdefault("output_text", str(result or ""))
            if not entry["response_text"]:
                entry["response_text"] = str(result or "")
            if entry["status"] == "running":
                entry["status"] = "received"
            return result
        except Exception as error:
            entry.update(status="error", error=str(error), error_type=type(error).__name__)
            raise
        finally:
            entry["seconds"] = round(time.monotonic() - started, 3)
            _active.reset(token)
            _changed()
    return wrapped
