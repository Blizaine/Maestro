"""Remote structured-output opt-in wrapper for H3 planning calls."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from services.h3_remote_schema import (  # noqa: E402
    structured_output_rejected,
    with_remote_structured_output,
)

SCHEMA = {"type": "object"}


class Recorder:
    def __init__(self, *outcomes):
        self.calls = []
        self.outcomes = list(outcomes)

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return "ok"


def _wrap(recorder, provider):
    return with_remote_structured_output(recorder, provider=lambda: provider)


@pytest.mark.parametrize("provider", ["remote", "openai"])
def test_opt_in_added_for_remote_providers(provider):
    rec = Recorder()
    assert _wrap(rec, provider)("prompt", json_schema=SCHEMA) == "ok"
    assert rec.calls == [(("prompt",), {"json_schema": SCHEMA, "remote_json_schema": True})]


@pytest.mark.parametrize("provider", ["local", "anthropic", None])
def test_no_opt_in_for_other_providers(provider):
    rec = Recorder()
    _wrap(rec, provider)("prompt", json_schema=SCHEMA)
    assert rec.calls == [(("prompt",), {"json_schema": SCHEMA})]


def test_no_opt_in_without_json_schema():
    rec = Recorder()
    gen = _wrap(rec, "remote")
    gen("prompt")
    gen("prompt", json_schema=None)
    assert rec.calls[0] == (("prompt",), {})
    assert rec.calls[1] == (("prompt",), {"json_schema": None})


def test_explicit_caller_value_is_preserved():
    rec = Recorder()
    _wrap(rec, "remote")("prompt", json_schema=SCHEMA, remote_json_schema=False)
    assert rec.calls == [(("prompt",), {"json_schema": SCHEMA, "remote_json_schema": False})]


def test_rejection_retries_without_opt_in_and_disables_later_calls():
    err = RuntimeError("LLM request failed: response_format json_schema is not supported")
    rec = Recorder(err, "retried", "second")
    gen = _wrap(rec, "remote")
    assert gen("p", json_schema=SCHEMA) == "retried"
    assert rec.calls[0][1].get("remote_json_schema") is True
    assert "remote_json_schema" not in rec.calls[1][1]
    assert gen("p", json_schema=SCHEMA) == "second"
    assert len(rec.calls) == 3
    assert "remote_json_schema" not in rec.calls[2][1]


def test_disable_is_per_wrapper_instance():
    err = RuntimeError("400 Client Error: Bad Request for url: https://x")
    first = Recorder(err, "x")
    _wrap(first, "remote")("p", json_schema=SCHEMA)
    second = Recorder()
    _wrap(second, "remote")("p", json_schema=SCHEMA)
    assert second.calls[0][1].get("remote_json_schema") is True


def test_unrecognized_error_is_reraised_without_retry():
    rec = Recorder(RuntimeError("LLM request failed: Read timed out"))
    with pytest.raises(RuntimeError, match="Read timed out"):
        _wrap(rec, "remote")("p", json_schema=SCHEMA)
    assert len(rec.calls) == 1


def test_interrupted_error_is_reraised_without_retry():
    rec = Recorder(InterruptedError("cancelled: invalid response_format schema"))
    with pytest.raises(InterruptedError):
        _wrap(rec, "remote")("p", json_schema=SCHEMA)
    assert len(rec.calls) == 1


def test_retry_failure_propagates():
    rec = Recorder(RuntimeError("422 Client Error: Unprocessable"), RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        _wrap(rec, "remote")("p", json_schema=SCHEMA)
    assert len(rec.calls) == 2


@pytest.mark.parametrize("message", [
    "response_format is unsupported",
    "Unknown parameter: json_schema",
    "Structured output not supported by this model",
    "Invalid schema for response_format 'x'",
    "400 Client Error: Bad Request for url: https://x",
    "422 Client Error: Unprocessable Entity",
    (
        "LLM request failed: 400 Client Error: Bad Request for url: "
        "https://x/v1/chat/completions\nEndpoint response: {\"error\": {\"message\": "
        "\"Invalid schema for response_format 'maestro_response'\"}}"
    ),
])
def test_structured_output_rejected_true(message):
    assert structured_output_rejected(RuntimeError(message))


@pytest.mark.parametrize("message", [
    "LLM request failed: Read timed out",
    "500 Server Error: Internal Server Error",
    "401 Client Error: Unauthorized",
    "Connection refused",
    "schema looks fine",
    "invalid api key",
])
def test_structured_output_rejected_false(message):
    assert not structured_output_rejected(RuntimeError(message))


def test_structured_output_rejected_false_for_interrupted():
    assert not structured_output_rejected(InterruptedError("400 Client Error"))
