"""Send H3 planning schemas to remote OpenAI-compatible providers.

The local llama-server already enforces every H3 planning schema as a
grammar. A remote provider receives the schema only when a call opts in with
``remote_json_schema=True``, and the H3 story/window planner never did. Remote
writers then had to infer each card's shape from prose and routinely dropped
required fields (for example ``sound_effects``), so every window failed its
contract check and fell back to source-compiled text.

``with_remote_structured_output`` opts every schema-bearing planner call in
while a remote provider is active. If the provider rejects strict structured
output, the call is retried exactly as before (unconstrained) and the opt-in is
switched off for the rest of that planning run, so a rejecting gateway costs at
most one extra request.
"""

from __future__ import annotations

import re
from typing import Any, Callable

_REMOTE_PROVIDERS = frozenset({"remote", "openai"})
_FORMAT_MARKERS = (
    "response_format", "json_schema", "json schema", "structured output", "schema",
)
_REJECTION_MARKERS = (
    "unsupported", "not supported", "unknown", "unrecognized", "invalid",
    "not accepted", "does not accept", "does not support", "rejected", "not allowed",
)
# requests' HTTPError text, e.g. "400 Client Error: Bad Request for url: ...".
_BAD_REQUEST = re.compile(r"\b(?:400|422) client error\b")


def _active_provider() -> str | None:
    from services import llm_service

    return getattr(llm_service, "_provider", None)


def structured_output_rejected(error: Exception) -> bool:
    """Return whether a remote request most likely failed on response_format.

    The opt-in is the only new field this wrapper adds, so a bare 400/422 from
    the gateway is treated as a rejection too; a request that was invalid for
    another reason simply fails again on the unconstrained retry.
    """

    if isinstance(error, InterruptedError):
        return False
    message = str(error).casefold()
    if any(marker in message for marker in _FORMAT_MARKERS) and any(
        marker in message for marker in _REJECTION_MARKERS
    ):
        return True
    return bool(_BAD_REQUEST.search(message))


def with_remote_structured_output(
    generate: Callable[..., str],
    *,
    provider: Callable[[], str | None] = _active_provider,
) -> Callable[..., str]:
    """Wrap ``llm_service.generate`` for one H3 planning run."""

    state = {"enabled": True}

    def structured_generate(*args: Any, **kwargs: Any) -> str:
        if (
            not state["enabled"]
            or kwargs.get("json_schema") is None
            or "remote_json_schema" in kwargs
            or provider() not in _REMOTE_PROVIDERS
        ):
            return generate(*args, **kwargs)
        try:
            return generate(*args, **kwargs, remote_json_schema=True)
        except Exception as error:
            if not structured_output_rejected(error):
                raise
            state["enabled"] = False
            print(
                "[MiniMax H3] Remote provider rejected strict JSON Schema; "
                f"continuing this plan without it. ({error})"
            )
            return generate(*args, **kwargs)

    return structured_generate
