"""Request identity shared by the HTTP boundary and job registration."""
from contextvars import ContextVar
from typing import Any

principal_context: ContextVar[Any] = ContextVar("maestro_agent_principal", default=None)
credential_context: ContextVar[str | None] = ContextVar("maestro_agent_credential", default=None)
submission_context: ContextVar[str | None] = ContextVar("maestro_agent_submission", default=None)
public_origin_context: ContextVar[str | None] = ContextVar("maestro_agent_public_origin", default=None)


def current_principal_id() -> str | None:
    principal = principal_context.get()
    return principal.id if principal is not None else None
