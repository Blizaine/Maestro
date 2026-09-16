"""Opt-in, request-scoped evidence; ordinary generation has no stored trace."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import time

_trace = ContextVar("promptbench_trace", default=None)
_call = ContextVar("promptbench_call", default=None)


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()
                if not any(part in k.lower() for part in ("api_key", "authorization", "password"))}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, str) and value.startswith("data:"):
        return {"data_url_sha256": hashlib.sha256(value.encode()).hexdigest(),
                "characters": len(value), "media_type": value.split(";", 1)[0][5:]}
    return value


@contextmanager
def capture(deadline_seconds=900, max_calls=32):
    record = {"calls": [], "deadline": time.monotonic() + deadline_seconds, "max_calls": max_calls}
    token = _trace.set(record)
    try:
        yield record
    finally:
        _trace.reset(token)


def expired():
    record = _trace.get()
    return bool(record and time.monotonic() >= record["deadline"])


def record_payload(payload, *, model_id=None, command=None):
    call = _call.get()
    if call is not None:
        call["payload"] = sanitize(payload)
        call["model_id"] = model_id
        call["runtime_command"] = sanitize(command)
    return payload


def record_metrics(metrics):
    call = _call.get()
    if call is not None:
        call["metrics"] = sanitize(metrics)


def record_response(content, reasoning="", *, complete=True):
    """Retain local-writer text before stripping, including an interrupted stream."""
    call = _call.get()
    if call is not None:
        call["raw_response"] = {"content": sanitize(content), "reasoning_content": sanitize(reasoning),
                                "complete": complete}


def traced(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        record = _trace.get()
        # Cancellable generate() delegates to generate_streaming(). They are
        # one request: let the inner transport fill the enclosing call record.
        if record is None or _call.get() is not None:
            return function(*args, **kwargs)
        if expired() or len(record["calls"]) >= record["max_calls"]:
            raise InterruptedError("Prompt bench request reached its time or LLM-call limit.")
        entry = {"index": len(record["calls"]) + 1, "function": function.__name__, "status": "running"}
        record["calls"].append(entry)
        token = _call.set(entry)
        started = time.monotonic()
        try:
            result = function(*args, **kwargs)
            entry.update(status="complete", output=sanitize(result))
            return result
        except Exception as error:
            entry.update(status="failed", error=str(error), error_type=type(error).__name__)
            raise
        finally:
            entry["seconds"] = round(time.monotonic() - started, 3)
            _call.reset(token)
    return wrapped
