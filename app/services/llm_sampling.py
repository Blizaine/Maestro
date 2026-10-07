"""Request-scoped sampling overrides for focused writing and fidelity repairs."""
from contextlib import contextmanager
from contextvars import ContextVar

_caller_sampling = ContextVar('llm_caller_sampling', default=False)


def caller_sampling_enabled() -> bool:
    return _caller_sampling.get()


@contextmanager
def caller_sampling(enabled: bool = False):
    """Honor a repair's temperature/top_p without changing model guard defaults."""
    token = _caller_sampling.set(enabled is True)
    try:
        yield
    finally:
        _caller_sampling.reset(token)
